"""Verify one retained affine prediction without loading a learner or source data.

The expected deployment ID, map hash, query ID, and reported value are supplied
by the caller's trusted context, not inferred from the certificate being tested.
"""
import argparse,json
from pathlib import Path
import numpy as np
from certificate41 import prepare_deployment,verify_certificate,CertificateError
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',required=True);p.add_argument('--features',required=True)
    p.add_argument('--certificate',required=True);p.add_argument('--query-id',required=True)
    p.add_argument('--expected-deployment-id',required=True);p.add_argument('--expected-map-sha256',required=True)
    p.add_argument('--reported-physical-hex',required=True);p.add_argument('--row',type=int,default=0)
    args=p.parse_args()
    deployment=json.loads(Path(args.deployment).read_text());packet=json.loads(Path(args.certificate).read_text())
    z=np.load(args.features,allow_pickle=False)
    if z.ndim==2:z=z[args.row]
    prepared=prepare_deployment(deployment,expected_deployment_id=args.expected_deployment_id,expected_map_sha256=args.expected_map_sha256)
    result=verify_certificate(prepared,z,packet,expected_query_id=args.query_id,reported_prediction=float.fromhex(args.reported_physical_hex))
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':
    try:main()
    except (CertificateError,ValueError,KeyError) as error:
        print(json.dumps({'status':'REJECT','reason':str(error)},sort_keys=True));raise SystemExit(2)
