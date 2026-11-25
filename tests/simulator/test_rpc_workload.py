"""Regression: workload labels survive the multiprocessing RPC boundary."""

import pickle
from unittest.mock import Mock

import pytest

from vllm import SamplingParams
from vllm.engine.multiprocessing import RPCProcessRequest
from vllm.engine.multiprocessing.engine import MQLLMEngine


@pytest.mark.parametrize("workload", [None, "coding-agent", "quality"])
def test_rpc_preserves_workload_through_engine_submission(workload):
    request = RPCProcessRequest(
        prompt="hello",
        params=SamplingParams(),
        request_id="rpc-label",
        workload=workload,
    )
    received = pickle.loads(pickle.dumps(request))
    backend = object.__new__(MQLLMEngine)
    backend.engine = Mock()
    backend._errored_with = None
    backend.log_requests = False
    backend._handle_process_request(received)
    assert backend.engine.add_request.call_count == 1
    assert backend.engine.add_request.call_args.kwargs["workload"] == workload
