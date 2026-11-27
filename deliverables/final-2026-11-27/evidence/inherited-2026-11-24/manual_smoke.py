"""Exercise the README's standalone preparation/API/replay path, not a benchmark."""
from pathlib import Path
import json,os,sys,socket,subprocess,time,urllib.request
R=Path(__file__).resolve().parents[2];repo=R/'work/review-vllm';out=R/'work/final-2026-11-24/manual-smoke';out.mkdir(exist_ok=False)
model='Qwen/Qwen2.5-0.5B-Instruct';rev='7ae557604adf67be50417f59c2c2f167def9a775'
prepare=[sys.executable,'-m','vllm.simulator.study','prepare','--dataset',str(R/'work/rerun-2026-11-23/data/sharegpt1000.jsonl'),'--output',str(out/'prepared'),'--model',model,'--revision',rev,'--limit','2','--max-model-len','4096','--mode','multi','--arrival','poisson','--request-rate','10']
subprocess.run(prepare,cwd=repo,check=True)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
server=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',model,'--revision',rev,'--tokenizer-revision',rev,'--distributed-executor-backend','sim','--load-format','dummy','--sim-trace-path',str(out/'prepared/trace.jsonl'),'--sim-strict-trace','--sim-metrics-path',str(out/'server/blocks.jsonl'),'--sim-kv-cache-gb','0.1','--max-model-len','4096','--block-size','16','--enable-prefix-caching','--enable-prompt-tokens-details','--host','127.0.0.1','--port',str(port)]
(out/'commands.json').write_text(json.dumps({'prepare':prepare,'server':server,'replay_equivalent':[sys.executable,'-m','vllm.simulator.study','replay','--workload',str(out/'prepared/workload.json'),'--base-url',f'http://127.0.0.1:{port}','--output',str(out/'requests.jsonl'),'--concurrency','2'],'frontend':'default multiprocessing; no disable flag','purpose':'standalone README correctness smoke, not an experimental configuration'},indent=2))
env=dict(os.environ);env.pop('VLLM_SIM_EVICTION_COUNTS_PATH',None)
with (out/'server.log').open('w') as log:
 child=subprocess.Popen(server,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+120
  while time.monotonic()<deadline:
   if child.poll() is not None:raise RuntimeError('Standalone API exited; inspect server.log')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
     if response.status==200:break
   except Exception:time.sleep(.2)
  else:raise TimeoutError('Standalone API startup timeout')
  workload=json.loads((out/'prepared/workload.json').read_text())
  replay_command=json.loads((out/'commands.json').read_text())['replay_equivalent']
  subprocess.run(replay_command,cwd=repo,env=env,check=True)
  rows=[json.loads(line) for line in (out/'requests.jsonl').read_text().splitlines() if line]
  expected=sum(len(c['turns']) for c in workload['conversations']);assert len(rows)==expected and all(r['success'] for r in rows)
  result={'purpose':'correctness smoke; excluded from experiment totals','conversations':len(workload['conversations']),'requests':len(rows),'failed':sum(not r['success'] for r in rows),'frontend':'multiprocessing default','weights_or_gpu_required':False};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 finally:
  child.terminate()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:child.kill();child.wait()
