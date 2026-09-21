# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared kernel-compilation core for PyPTO model executors."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .l3_callable import L3Callable

if TYPE_CHECKING:
    from pypto.runtime import RunConfig

logger = logging.getLogger(__name__)


class KernelCompiler:
    """Compile HOST kernels through PyPTO's validated persistent JIT cache.

    PyPTO owns specialization keys, source/toolchain identity, immutable
    publication and restoration. Ordinary DistributedWorker preparation builds
    and publishes READY binaries; serving never reloads a directory by name.

    Serving sets no cache policy of its own: the effective policy is whatever
    ``RunConfig.cache_config`` carries (normally ``None``), and PyPTO resolves
    a ``None`` policy from ``pypto.configure_cache()`` and the
    ``PYPTO_CACHE`` / ``PYPTO_CACHE_DIR`` / ``PYPTO_CACHE_READONLY``
    environment of each worker process. Diagnostic/output controls retain
    PyPTO's bypass rules.
    """

    def __init__(
        self,
        *,
        run_config: "RunConfig",
        **extra_configs: Any,
    ) -> None:
        self._run_config = run_config
        self._extra_configs = extra_configs

    def compile(
        self,
        name: str,
        jit_fn: object,
        **compile_kwargs: Any,
    ) -> L3Callable:
        """Compile in signature mode and retain runtime scalar keyword arguments.

        The compile ``RunConfig`` carries ``cache_config`` untouched: either
        the executor's explicit policy or ``None``, which PyPTO resolves
        through its own process and ``PYPTO_CACHE*`` environment handling.
        """
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        configs = {**self._extra_configs, "codegen_only": True}
        if self._run_config.save_kernels_dir is not None and "save_kernels_dir" not in configs:
            configs["save_kernels_dir"] = str(Path(self._run_config.save_kernels_dir) / name)

        # Preserve every compiler/diagnostic field, including future RunConfig
        # additions. In particular, an explicit save_kernels_dir still bypasses
        # reuse; serving's normal build directory is configured separately.
        run_config = dataclasses.replace(self._run_config, **configs)
        # Worded so log-marker contracts (e.g. the DSpark K=7 one-L2 guard)
        # keep matching: PyPTO itself decides compile vs. validated restore.
        logger.info("[kernel-compile] compiling %s through PyPTO JIT", name)
        compiled = jit_fn.compile(config=run_config, **compile_kwargs)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; "
                f"got {type(compiled).__name__}"
            )
        return L3Callable(
            compiled=compiled,
            name=name,
            aicpu_thread_num=run_config.distributed_config.aicpu_thread_num,
        )
