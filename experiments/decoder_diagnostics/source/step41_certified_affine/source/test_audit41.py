"""Independent-checker artificial/adversarial tests; no public data is decoded."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import audit41 as a
import certificate41 as c
import affine41

CHECKS=[]
def check(name,value):
    CHECKS.append({'name':name,'pass':bool(value)})
    if not value:raise AssertionError(name)
def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,TypeError,OverflowError):check(name,True);return
    check(name,False)
def altered(value,key,new):
    value=deepcopy(value);value[key]=new;return value

def main():
    head={'schema':'step35-stored-feature-head-v1','kind':'clipped_affine','feature_id':'synthetic-feature41',
          'dimension':2,'kappa':'2','output_interval':['0','1'],'assurance':a.ASSURANCE,
          'coefficients':['1/3','-1/5'],'intercept':'1/7'}
    head['class_id']=a.class_identity(head['feature_id'])
    dp=c.make_deployment(head,target_min='-23/10',span='17/3',map_sha256='a'*64,map_bytes=1234,candidate_id='synthetic::affine')
    prepared=c.prepare_deployment(dp);z=np.asarray([.7,-.2],dtype=np.float64)
    reported=c.predict_float(prepared,z);packet=c.produce_certificate(prepared,z,reported,query_id='synthetic:q0')
    checked=a.audit_certificate(dp,z,packet,query_id='synthetic:q0',reported_prediction=reported)
    check('independent_fraction_matches_integer_reference',checked['exact_reference_physical']==c.verify_certificate(prepared,z,packet,expected_query_id='synthetic:q0',reported_prediction=reported)['exact_reference_physical'])
    check('independent_feature_packet_bytes',a.feature_packet(z)==c.feature_payload_bytes(z))
    check('static_valid_deployment',a.audit_deployment(dp)['status']=='PASS')
    for label,vector in [('lower_clip',[-100.,100.]),('upper_clip',[100.,-100.]),('signed_zero',[0.,-0.]),('subnormal',[math.ulp(0.),-math.ulp(0.)])]:
        zz=np.asarray(vector,dtype=np.float64);value=c.predict_float(prepared,zz);w=c.produce_certificate(prepared,zz,value,query_id=label)
        check(label+'_independent_arithmetic',a.audit_certificate(dp,zz,w,query_id=label,reported_prediction=value)['status']=='PASS')
    for key,value in [('query_id','wrong'),('deployment_id','deploy41:'+'b'*64),('feature_sha256','b'*64),
                      ('reported_prediction_hex',math.nextafter(reported,math.inf).hex()),
                      ('reference_interval_hex',[0.0.hex(),100.0.hex()]),('absolute_error_upper_hex',0.0.hex()),
                      ('schema','step00-invalid')]:
        reject('reject_certificate_'+key,lambda key=key,value=value:a.audit_certificate(dp,z,altered(packet,key,value),query_id='synthetic:q0',reported_prediction=reported))
    reject('reject_wrong_external_query',lambda:a.audit_certificate(dp,z,packet,query_id='other',reported_prediction=reported))
    reject('reject_wrong_external_prediction',lambda:a.audit_certificate(dp,z,packet,query_id='synthetic:q0',reported_prediction=reported+.1))
    reject('reject_changed_features',lambda:a.audit_certificate(dp,z[::-1],packet,query_id='synthetic:q0',reported_prediction=reported))
    reject('reject_extra_packet_field',lambda:a.audit_certificate(dp,z,{**packet,'future_coverage':True},query_id='synthetic:q0',reported_prediction=reported))
    bad=reported+1.;consistent=c.produce_certificate(prepared,z,bad,query_id='synthetic:q0')
    reject('reject_self_consistent_large_prediction_error',lambda:a.audit_certificate(dp,z,consistent,query_id='synthetic:q0',reported_prediction=bad))
    reject('reject_nan_hex',lambda:a.float_hex('nan'))
    reject('reject_noncanonical_hex',lambda:a.float_hex('1.0'))
    reject('reject_boolean_rational',lambda:a.q(True))
    reject('reject_nan_rational',lambda:a.q(float('nan')))
    reject('reject_false_map_identity',lambda:a.audit_deployment(dp,expected_map_sha256='b'*64))
    reject('reject_false_candidate_identity',lambda:a.audit_deployment(dp,expected_candidate='other'))
    reject('reject_false_map_byte_count',lambda:a.audit_deployment(dp,expected_map_bytes=1235))
    reject('reject_changed_scale_without_new_identity',lambda:a.audit_deployment(altered(dp,'span','2')))
    badhead=deepcopy(head);badhead['coefficients']=['3','0'];reject('reject_lipschitz_norm_violation',lambda:a.audit_head(badhead))
    badhead=deepcopy(head);badhead['class_id']='wrong';reject('reject_false_class_identity',lambda:a.audit_head(badhead))
    # Reconstruct the inherited fitted affine recipe only on artificial features.
    X=np.asarray([[float((i*7+j*13)%29)/29 for j in range(3)] for i in range(40)],dtype=np.float64)
    y=[F((i*11)%37,37) for i in range(40)]
    fitted=affine41.fit_affine(X,y,feature_id='synthetic-fit41')
    check('independent_affine_fit_projection_reconstruction',a.audit_affine_record(fitted,X,y)['status']=='PASS')
    badfit=deepcopy(fitted);badfit['projection']['center_target']='100'
    reject('reject_changed_training_center',lambda:a.audit_affine_record(badfit,X,y))
    badfit=deepcopy(fitted);badfit['projection']['factor']='1/2'
    reject('reject_changed_projection_factor',lambda:a.audit_affine_record(badfit,X,y))
    reject('reject_different_training_labels',lambda:a.audit_affine_record(fitted,X,[v+1 for v in y]))
    check('primary_exact_boundary_accepted',a.audit_primary('21/20','1')['within_five_percent_mse'])
    check('primary_above_boundary_rejected',not a.audit_primary('10500000000000000001/10000000000000000000','1')['within_five_percent_mse'])
    check('primary_zero_reference_not_established',not a.audit_primary('0','0')['within_five_percent_mse'])
    check('five_percent_improvement_exact_boundary',a.audit_primary('19/20','1')['at_least_five_percent_lower_mse'])
    reject('negative_mse_rejected',lambda:a.audit_primary('-1','1'))
    loss=a.physical_loss([F(1,3),F(2,3)],['0','1'])
    check('exact_physical_squared_loss',loss=={'n':2,'sse':'2/9','mse':'1/9'})
    rows=[{'candidate_id':'b','pooled_physical_mse':'1','selection_payload_bytes':30},
          {'candidate_id':'a','pooled_physical_mse':'1','selection_payload_bytes':30},
          {'candidate_id':'c','pooled_physical_mse':'1','selection_payload_bytes':29}]
    check('pooled_tie_uses_size',a.audit_pooled_selection(rows,'synthetic',[r['candidate_id'] for r in rows])['candidate_id']=='c')
    check('pooled_tie_uses_lexical',a.audit_pooled_selection(rows[:2],'synthetic',['b','a'])['candidate_id']=='a')
    rows[0]['pooled_physical_mse']='999999999999999999/1000000000000000000'
    check('exact_risk_precedes_size',a.audit_pooled_selection(rows,'synthetic',['a','b','c'])['candidate_id']=='b')
    reject('repeated_selection_id_refused',lambda:a.audit_pooled_selection(rows+rows[:1],'synthetic',['a','b','c']))
    reject('empty_selection_pool_refused',lambda:a.audit_pooled_selection(rows,'synthetic',[]))
    check('independent_csv_quotes',a.csv_cells(b'"a,b",c,"d""e"\r\n')==[b'a,b',b'c',b'd"e'])
    reject('malformed_csv_quotes_refused',lambda:a.csv_cells(b'"a,b,c\n'))
    require_scope=a.ARTIFICIAL_FIXTURE
    a.ARTIFICIAL_FIXTURE=True
    reject('fixture_cannot_target_scientific_root',a.artificial_scope)
    a.ARTIFICIAL_FIXTURE=require_scope
    receipt={'schema':'step41-independent-audit-artificial-checks-v1','status':'PASS','checks':CHECKS,
             'count':len(CHECKS),'public_measurement_values_decoded':0,
             'audit_source_sha256':a.sha(Path(a.__file__)),'test_source_sha256':a.sha(Path(__file__))}
    a.write(a.ROOT/'audit/artificial_checks41.json',receipt)
    design={'schema':'step41-independent-audit-design-v1','status':'READY_FOR_FROZEN_RUN','checks':[
      'Independent byte/timestamp/missing-status source projection before numerical access',
      'Exactly authorized source decimals and causal24hour history after completed access receipt',
      'Training-only map/scaler/targetnormalization and inherited affine fitting/projection',
      'Independent Fraction evaluation of compact deployment certificates and exact errorbudget',
      'Operational affine/baseline prediction replay and exact physical SSE/MAE arithmetic',
      'Pooled developmentselection, deterministictie-breaking and fixedtest criterion',
      'Perquery prediction seals precedingcurrenttargetreads and globalexecution hashchain',
      'Selecteddeployedmodel cost/packet accounting and independent summaryrecalculation'],
      'artificial_checks_path':'audit/artificial_checks41.json','artificial_checks_sha256':a.sha(a.ROOT/'audit/artificial_checks41.json'),
      'producer_math_or_risk_or_selection_helpers_imported':False,
      'artificial_checks':len(CHECKS),'public_measurement_values_decoded':0,
      'limits':['Stored-feature mathematical guarantee does not enclose native transform arithmetic',
                'Baseline library prediction replay checks stored payload, not a theorem of global training optimality',
                'Local byte seals/event chains are provenance, not an externally trusted timestamp',
                'Timing source and summaries are checked; hardware clock measurements are not independently attested']}
    a.write(a.ROOT/'analysis/audit_design41.json',design)
    print(json.dumps({'status':'PASS','checks':len(CHECKS),'public_measurement_values_decoded':0}))

if __name__=='__main__':main()
