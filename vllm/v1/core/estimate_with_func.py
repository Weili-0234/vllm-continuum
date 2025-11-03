from vllm.v1.request import Request
from typing import Optional
import time
from vllm.logger import init_logger
logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds

class Continuum_Recorder:
    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing
        self.scheduling_times = []  # List of {start_time, end_time, duration}

    def print_history(self):
        import os
        import json

        # Per-run output directory (set by launcher); fallback to default
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")
        os.makedirs(output_dir, exist_ok=True)

        # Atomic write to avoid partial reads by other processes
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(tmp_path, final_path)

        # Also save scheduling timing statistics
        if self.scheduling_times:
            scheduling_stats_path = os.path.join(output_dir, "scheduling_timing_stats")
            tmp_stats_path = scheduling_stats_path + ".tmp"
            
            # Calculate statistics
            durations = [s["duration"] for s in self.scheduling_times]
            stats = {
                "total_scheduling_calls": len(self.scheduling_times),
                "total_scheduling_time": sum(durations),
                "average_scheduling_time": sum(durations) / len(durations),
                "min_scheduling_time": min(durations),
                "max_scheduling_time": max(durations),
                "scheduling_times": self.scheduling_times
            }
            
            with open(tmp_stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            os.replace(tmp_stats_path, scheduling_stats_path)

    def request_arrives(self, request: Request):
        if request.job_id not in self.job_id_to_history:
            self.job_id_to_history[request.job_id] = []
        self.job_id_to_history[request.job_id].append({"Request_arrival_time": time.time()})
    
    def request_finished(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_departure_time": time.time()})

    def request_evicted_from_running_queue(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"pinned_time": time.time()})

    def request_unpinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int, hit_length: int = 0):
        self.job_id_to_history[request.job_id].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })
    
    def request_evicted_to_running(self, request: Request, prompt_length: int, hit_length: int):
        self.job_id_to_history[request.job_id].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })
    
    def scheduling_started(self) -> float:
        """Mark the start of a scheduling operation and return the start time."""
        start_time = time.time()
        return start_time
    
    def scheduling_finished(self, start_time: float) -> None:
        """Mark the end of a scheduling operation and record the duration."""
        end_time = time.time()
        duration = end_time - start_time
        self.scheduling_times.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        })
    
class ToolCallEstimator:
    def __init__(self):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}

        self.job_to_history: dict[str, list[dict[str, float]]] = {}

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]
    
    #TODO Hanchen This is currently just an average 
    def update_func_call_exec_time(self, job_id: str) -> None:
        #this is called when the func call is back again in scheduler.py, update the exec time with last_func_call
        last_departure_time = self.job_to_history[job_id][-1]["departure_time"]
        func = self.job_to_history[job_id][-1]["func_call"]
        exec_time = time.time() - last_departure_time

        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)
        self.func_call_to_exec_time[func] = sum(self.record_func_call_to_exec_time[func]) / len(self.record_func_call_to_exec_time[func])
        return 
    
    #Functions below will be called by outside functions
    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return 0
        
        this_func_call_exec_time = self.get_func_call_exec_time(request.this_func_call) or 0.0

        if this_func_call_exec_time > FIXED_THRESHOLD_CONTINUUM:
            return 0
        
        return FIXED_THRESHOLD_CONTINUUM

    def request_arrives(self, request: Request) -> None:
        logger.info(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        # this is called when a job arrives in scheduler.py, if job is new, create an entry,
        if request.job_id not in self.job_to_history:
            logger.info(f"Request job id: {request.job_id}")
            self.job_to_history[request.job_id] = []
            assert request.last_func_call is None
            self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
            return
        logger.info(f"Request job id: {request.job_id}")
        # Else if there is a last_func_call, then call update_func_call_exec_time to update func_call_to_exec_time
        print(f"Request last_func_call: {request.last_func_call}")
        print(f"Job to history last func call: {self.job_to_history[request.job_id][-1]['func_call']}")
        assert request.last_func_call == self.job_to_history[request.job_id][-1]["func_call"]

        self.update_func_call_exec_time(request.job_id)

        self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
        return
    
    def request_finished(self, request: Request) -> None:
        logger.info(f"Request job id finishing: {request.job_id}, time is {time.time()}")   
        self.job_to_history[request.job_id].append({"departure_time": time.time(), "func_call": request.this_func_call})
        return

