from vllm.v1.core.estimate_with_func import ToolCallEstimator
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
import time
estimator = ToolCallEstimator()

#Time stamp1
request = Request(
    job_id="1",
    arrival_time=time.time(),
    last_func_call=None,
    this_func_call="big",
    is_last_step=False,
    sampling_params=SamplingParams(
        max_tokens=100
    )
) 

estimator.request_arrives(request)

time.sleep(1)

#Time stamp2
estimator.request_finished(request)

time.sleep(2)

request = Request(
    job_id="1",
    arrival_time=time.time(),
    last_func_call="big",
    this_func_call=None,
    is_last_step=True,
    sampling_params=SamplingParams(
        max_tokens=100
    )
) 
estimator.request_arrives(request)

time.sleep(1)
estimator.request_finished(request)

print(estimator.get_func_call_exec_time("big"))
print(estimator.get_func_call_time_until_finish("big"))