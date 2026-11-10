"""Generate an offline, interactive HTML experiment ledger from run directories."""
import argparse
import csv
import json
import os
from pathlib import Path
from plv_tracking import now, training_dir


def generate(root, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / 'index.html'
    records = []
    for meta in sorted(root.rglob('evaluation.json')):
        evaluation = json.loads(meta.read_text(encoding='utf-8'))
        run = meta.parent.parent.parent
        train = training_dir(run)
        config_path = train / 'run.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        summary = meta.parent / f"{evaluation['split']}_summary.csv"
        if evaluation.get('status') != 'completed' or not summary.exists():
            scores = []
        else:
            with summary.open(encoding='utf-8', newline='') as stream:
                scores = list(csv.DictReader(stream))
        records.append(dict(run=str(run.relative_to(root)), evaluation=meta.parent.name,
                            config=config, meta=evaluation, scores=scores,
                            report=os.path.relpath(summary, output.parent).replace(os.sep, '/')))
    # JSON 中的 '<' 必须转义，防止目录名/备注中的 </script> 提前关闭脚本。
    data = json.dumps(records, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    html = TEMPLATE.replace('__RECORDS__', data).replace('__GENERATED__', now())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    return output


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BOLD → PLV 实验台账</title>
<style>
body{font:15px system-ui,sans-serif;background:#f3f6fa;color:#182a40;margin:0;padding:28px}main{max-width:1500px;margin:auto}h1{margin:0 0 10px}p{color:#53657a;line-height:1.6}.panel{background:white;padding:20px;border:1px solid #dce5f0;border-radius:12px;margin:18px 0}label{display:inline-flex;gap:8px;align-items:center;margin:6px 18px 6px 0}select,input{padding:8px;border:1px solid #b8c9db;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px;border-bottom:1px solid #e5ebf3;vertical-align:top}th{background:#edf3fa;cursor:pointer;white-space:nowrap}small{color:#65758a}button{padding:9px 14px;cursor:pointer}a{color:#205cb6}.scroll{overflow:auto}.barrow{display:grid;grid-template-columns:minmax(180px,35%) 1fr 100px;gap:10px;align-items:center;margin:12px 0}.track{height:18px;background:#eef2f7;position:relative}.bar{height:18px;background:#417add;position:absolute}.missing{color:#aa4a32}details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.badge{display:inline-block;padding:3px 8px;background:#e8f0fc;border-radius:8px}
</style><main>
<h1>BOLD → EEG PLV 实验台账</h1>
<p>离线 HTML，无外部服务或网络依赖。生成时间：__GENERATED__。运行训练或评测后自动刷新文件；已打开的页面需要重新加载。</p>
<div class="panel"><label>筛选 <input id="search" placeholder="实验名 / 模型 / 备注"></label><label>划分 <select id="split"><option>test</option><option>val</option></select></label><label>频段 <select id="band"></select></label><label>指标 <select id="metric"></select></label><label><input type="checkbox" id="latest" checked>每实验仅显示最新评测</label><button id="download">导出当前表 CSV</button><p id="count"></p>
<p>每行是一份独立评测；默认展示测试集最终对比，可切换验证集查看开发记录。数据清单版本不同的实验只能作探索性比较。相关指标需同时检查有效比例；CI 为被试 bootstrap 区间。</p></div>
<div class="panel scroll"><table><thead><tr><th>实验 / 评测</th><th>模型 / seed / 采样</th><th>训练版本</th><th>数据版本</th><th id="valueTitle">指标</th><th>95% CI / 被试数</th><th>有效比例</th><th>状态 / 文件</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="panel"><h2>指标对比</h2><p id="direction"></p><div id="chart"></div><p>条形图以 0 为基准；详细数值、置信区间及有效被试数见上表。</p></div>
<div class="panel"><h2>运行配置与评测溯源</h2><div id="details"></div></div>
<script type="application/json" id="data">__RECORDS__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const metrics=['mse','mae','temporal_r','spatial_r','skill','delta_mse','template_mse','f1_0.25','f1_0.5','spatial_valid','temporal_valid'];
const bands=[...new Set(data.flatMap(r=>r.scores.map(s=>s.band)))];
function option(el,value){let o=document.createElement('option');o.value=value;o.textContent=value;el.append(o)}
(bands.length?bands:['delta','theta','alpha','beta','gamma','broadband']).forEach(v=>option($('band'),v));metrics.forEach(v=>option($('metric'),v));
function num(v){if(v===undefined||v===null||v==='')return null;const n=Number(v);return Number.isFinite(n)?n:null}
function fmt(v){const n=num(v);return n===null?'—':n.toFixed(5)}
function score(r,key){return r.scores.find(s=>s.band===$('band').value&&s.metric===key)}
function visible(){return data.filter(r=>r.meta.split===$('split').value&&(!$('latest').checked||!data.some(o=>o.run===r.run&&o.meta.split===r.meta.split&&o.meta.status==='completed'&&o.evaluation>r.evaluation))&&JSON.stringify([r.run,r.config.model,r.config.notes]).toLowerCase().includes($('search').value.toLowerCase())).sort((a,b)=>{let x=num(score(a,$('metric').value)?.mean),y=num(score(b,$('metric').value)?.mean);return x===null?1:y===null?-1:lower()?x-y:y-x})}
function lower(){return ['mse','mae','template_mse'].includes($('metric').value)}
function cell(row,text){let c=document.createElement('td');c.textContent=text;row.append(c);return c}
function render(){const list=visible(),key=$('metric').value;$('rows').replaceChildren();$('chart').replaceChildren();$('details').replaceChildren();$('valueTitle').textContent=key+(lower()?' ↓':' ↑');$('count').textContent=`共 ${data.length} 份评测，当前显示 ${list.length} 份。`;
 const versions=new Set(list.map(r=>r.meta.data_identity?.manifest_sha256||r.config.data_identity?.manifest_sha256||'unknown'));
 $('direction').textContent=(lower()?'数值越低越好。':'数值越高越好。')+(versions.size>1?' 注意：当前含不同数据清单版本。':'');
 const values=list.map(r=>num(score(r,key)?.mean)).filter(v=>v!==null),lo=Math.min(0,...values),hi=Math.max(0,...values),span=hi-lo||1;
 for(const r of list){const s=score(r,key)||{},tr=document.createElement('tr');cell(tr,r.run+'\n'+r.evaluation);cell(tr,(r.config.model||'?')+' / '+r.config.seed+' / '+(r.config.sampling||'global_random'));cell(tr,(r.config.provenance?.commit||'未记录').slice(0,12)+(r.config.provenance?.status?' (dirty)':''));cell(tr,(r.meta.data_identity?.manifest_sha256||r.config.data_identity?.manifest_sha256||'未记录').slice(0,12));cell(tr,fmt(s.mean));cell(tr,`[${fmt(s.ci_low)}, ${fmt(s.ci_high)}] / ${s.valid_subjects||'—'}`);
 cell(tr,key==='temporal_r'?fmt(score(r,'temporal_valid')?.mean):key==='spatial_r'?fmt(score(r,'spatial_valid')?.mean):'—');const c=cell(tr,r.meta.status+' ');if(r.meta.status==='completed'){let a=document.createElement('a');a.textContent='结果 CSV';a.href=r.report;c.append(a)}$('rows').append(tr);
 const v=num(s.mean);if(v!==null){let row=document.createElement('div');row.className='barrow';let label=document.createElement('span');label.textContent=r.run+' / '+r.evaluation;let track=document.createElement('div');track.className='track';let bar=document.createElement('div');bar.className='bar';bar.style.left=((Math.min(0,v)-lo)/span*100)+'%';bar.style.width=(Math.abs(v)/span*100)+'%';bar.style.background=v<0?'#c75c52':'#417add';track.append(bar);let value=document.createElement('span');value.textContent=fmt(v);row.append(label,track,value);$('chart').append(row)}
 let d=document.createElement('details'),title=document.createElement('summary'),pre=document.createElement('pre');title.textContent=r.run+' / '+r.evaluation;pre.textContent=JSON.stringify({training:r.config,evaluation:r.meta},null,2);d.append(title,pre);$('details').append(d);
 }if(!list.length)$('chart').textContent='暂无匹配的评测。完成一次训练或评测后重新生成台账。';}
['search','split','band','metric','latest'].forEach(id=>$(id).addEventListener('input',render));
$('download').onclick=()=>{const cols=['run','evaluation','model','seed','split','band','metric','mean','ci_low','ci_high','valid_subjects'];const rows=visible().map(r=>{const s=score(r,$('metric').value)||{};return [r.run,r.evaluation,r.config.model,r.config.seed,r.meta.split,$('band').value,$('metric').value,s.mean,s.ci_low,s.ci_high,s.valid_subjects]});const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const blob=new Blob(['\ufeff'+[cols,...rows].map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='plv_comparison.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></main></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--output')
    args = parser.parse_args()
    print(generate(args.runs, args.output))
