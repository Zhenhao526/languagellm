"""Bind a display-only analysis revision to the prior full independent audit."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/receiver_001'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    checks=[];failures=[]
    def check(ok,name):
        checks.append(name)
        if not ok:failures.append(name)
    audit=read(BATCH/'audit_analysis_independent.json')
    previous_json=BATCH/'analysis_versions/receiver_analysis_fa54493153ea.json'
    previous_source=BATCH/'analysis_versions/analyze_receiver_328c4de8cd84.py'
    current_json=BATCH/'receiver_analysis.json';current_source=ROOT/'analyze_receiver.py'
    old,new=read(previous_json),read(current_json)
    check(audit['passed'],'full_probability_and_seed_audit_passed')
    check(sha(previous_json)==audit['analysis_sha256'],'old_exact_JSON_matches_completed_audit')
    check(sha(previous_source)==audit['analysis_source_sha256'],'old_exact_analyzer_matches_completed_audit')
    check(sha(ROOT/'audit_receiver.py')==audit['audit_script_sha256'],'completed_analysis_auditor_unchanged')
    changed=[k for k in set(old)|set(new) if old.get(k)!=new.get(k)]
    check(changed==['analysis_sha256'],'JSON_only_analyzer_source_SHA_changed')
    check(old['analysis_sha256']==sha(previous_source) and new['analysis_sha256']==sha(current_source),
          'each_JSON_names_its_actual_analyzer')
    frozen=read(BATCH/'invocation.json')['source_hashes']
    for path,digest in frozen.items():check(sha(path)==digest,'frozen_training_and_metric_source_'+Path(path).name)
    before=ast.parse(previous_source.read_text());after=ast.parse(current_source.read_text())
    old_functions={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_functions={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    check(old_functions.keys()==new_functions.keys(),'same_function_inventory')
    for name in old_functions:
        if name!='figures':
            check(ast.dump(old_functions[name])==ast.dump(new_functions[name]),'nonfigure_AST_exact_'+name)
    replacements={
        'Common 24 maps: native sender + stochastic receiver':'Common 24 maps: native sender + random receiver',
        'Stochastic receiver joint success Q':'Random-receiver joint success Q',
        'Stochastic receiver mixed reward':'Random-receiver mixed reward',
        'Native sender + stochastic receiver: Q (%)':'Native sender + random receiver: Q (%)'}
    expected_additions=ast.parse("fig.get_layout_engine().set(rect=(0,.055,1,.945))\nfig.text(.5,.012,'* CE30: all 30 maps included in supervised fitting.',ha='center',va='bottom',fontsize=8)").body
    figure=new_functions['figures']
    save=next(n for n in figure.body if isinstance(n,ast.FunctionDef) and n.name=='save')
    check(all(ast.dump(a)==ast.dump(b) for a,b in zip(save.body[:2],expected_additions)),
          'only_declared_layout_and_CE30_footnote_statements_added')
    save.body=save.body[2:]
    class UndoLabels(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in replacements:
                node.value=replacements[node.value]
            return node
    figure=UndoLabels().visit(figure)
    check(ast.dump(old_functions['figures'])==ast.dump(figure),'all_plotted_values_and_plot_logic_AST_unchanged')
    # Whole-module equality catches changes to constants, imports, or main entry as well.
    check(ast.dump(before)==ast.dump(after),'whole_analyzer_AST_only_declared_display_revision')
    revision=read(BATCH/'figure_label_revision.json')
    check(revision['current_analysis_json_sha256']==sha(current_json) and
          revision['current_analysis_source_sha256']==sha(current_source) and
          revision['numerical_fields_changed'] is False and revision['metrics_recomputed'] is False,
          'label_revision_receipt_matches_independent_JSON_and_AST_checks')
    qa=read(BATCH/'figure_qa.json')
    check(qa['status']=='passed' and qa['analysis_sha256']==sha(current_json) and qa['script_sha256']==sha(current_source),
          'figure_QA_bound_to_current_revision')
    check(qa['report_sha256']==sha(BATCH/'接收诊断分析草稿.md'),'unchanged_draft_report_matches_QA')
    for figure in qa['figures']:
        check(sha(figure['png'])==figure['png_sha256'] and sha(figure['pdf'])==figure['pdf_sha256'],
              'current_PNG_PDF_fingerprints_'+Path(figure['png']).stem)
    def count_scalars(value):
        if isinstance(value,dict):return sum(count_scalars(x) for x in value.values())
        if isinstance(value,list):return sum(count_scalars(x) for x in value)
        return int(isinstance(value,(float,int)) and not isinstance(value,bool))
    result=dict(passed=not failures,checks=checks,total_checks=len(checks),failures=failures,
        unchanged_numeric_scalars=count_scalars(old),full_analysis_audit_sha256=sha(BATCH/'audit_analysis_independent.json'),
        previously_audited_JSON_sha256=sha(previous_json),current_JSON_sha256=sha(current_json),
        previous_analyzer_sha256=sha(previous_source),current_analyzer_sha256=sha(current_source),
        script_sha256=sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent equality of every JSON field except the analyzer SHA and exact AST check limited to declared plotting-label/layout changes. Binds the prior full numeric audit to the current display revision; no formula recomputation, inference or training.')
    output=BATCH/'audit_display_revision.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),unchanged_numeric_scalars=result['unchanged_numeric_scalars'],failures=failures,output=str(output))))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
