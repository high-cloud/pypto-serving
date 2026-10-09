# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep the Qwen3 device guard's instrumentation compatible with serving APIs."""

from types import SimpleNamespace

import pytest

from pypto_serving.model import tokenizer as tokenizer_module
from pypto_serving.serving.engine import async_engine
from tests import test_qwen3_serving as guard


@pytest.fixture
def observed_harness(monkeypatch, tmp_path):
    tokenizer = object()
    loads = []
    calls = []
    lifecycle = []
    output = SimpleNamespace(scheduled_requests=[SimpleNamespace(
        request=SimpleNamespace(request_id="request"), is_prefill=True, num_new_tokens=4,
    )])

    def load_tokenizer(model_dir):
        loads.append(model_dir)
        return tokenizer

    def schedule(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    async def start():
        lifecycle.append("start")

    async def stop():
        lifecycle.append("stop")

    def make_engine(*, config, tokenizer):
        return SimpleNamespace(
            tokenizer=tokenizer,
            scheduler=SimpleNamespace(
                config=SimpleNamespace(long_prefill_token_threshold=config.long_prefill_token_threshold),
                schedule=schedule,
            ),
            kv_cache_manager=SimpleNamespace(get_computed_blocks=lambda tokens: []),
            _cores=[SimpleNamespace(_async_scheduling=True)],
            start=start,
            stop=stop,
        )

    monkeypatch.setattr(guard, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(guard, "DEVICE_ID", 0)
    monkeypatch.setattr(tokenizer_module, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(async_engine, "AsyncLLMEngine", make_engine)
    fixture = guard.harness.__wrapped__()
    harness = next(fixture)
    try:
        yield harness, tokenizer, loads, calls, output, tmp_path
    finally:
        fixture.close()
        assert lifecycle == ["start", "stop"]
        assert harness.loop.is_closed()


def test_guard_uses_shared_tokenizer_loader(observed_harness):
    harness, tokenizer, loads, _, _, model_dir = observed_harness
    assert loads == [str(model_dir)]
    assert harness.engine.tokenizer is tokenizer


@pytest.mark.parametrize("options", [{}, {"allow_reject_stalled": False}, {"allow_reject_stalled": True}])
def test_guard_forwards_scheduler_options(observed_harness, options):
    harness, _, _, calls, output, _ = observed_harness
    assert harness.engine.scheduler.schedule(**options) is output
    assert calls == [((), options)]
    assert harness.schedule_events == [[("request", True, 4)]]
