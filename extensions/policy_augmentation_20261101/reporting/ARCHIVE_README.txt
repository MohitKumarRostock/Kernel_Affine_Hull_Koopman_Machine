KAHM POLICY AUGMENTATION: VERIFIED RESEARCH ARCHIVE

Contents
--------
KAHM_Policy_Augmentation_Analysis.zip contains the complete analysis tables, study protocols,
configuration, roster/provenance metadata, passed production audits, source
freezes, fitted/prediction seals, and reporting metrics available at packaging.
KAHM_Policy_Augmentation_Records_01.zip through Records_18.zip contain ALL generated
prepared arrays and targets, fitted states, B/C forecasts, basis records,
terminal forecasts, logs, and the natural study's official Train/metadata CSVs.
Each part is an ordinary independent ZIP containing complete files. No
byte-concatenation or multipart ZIP software is needed. Restore all parts to
reproduce the complete experiment; a single part is not the whole study.

The final PDF report is distributed separately. Source code and the complete
Git repository are not duplicated in these archives. Unsealed cache,
temporary, and manufactured fixture files are excluded and listed in the
manifest. Every file explicitly named by a scientific seal is retained,
including noncanonical filenames, so those exact seals remain verifiable.

Scientific source
-----------------
Repository: https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine
Branch: research/kahm-policy-augmentation-2026-11-01
Combined scientific source commit: b1915813e98a59572a4bbb800adc2bd1f3fccc1b
Earlier synthetic-only source publication: e41b6e4fb902e49c8438f1adf2a3e98cb429cddd

The immutable combined source commit contains BOTH study runners and frozen
configurations. Its natural-source publication occurred after that evaluation
completed. The local pre-evaluation source freeze documents its actual
chronology; this is NOT a public preregistration. No prior-source date is
retrospectively substituted for the actual natural publication.

Restore into a fresh checkout
----------------------------
git clone https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine.git kahm-policy-reproduction
cd kahm-policy-reproduction
git checkout b1915813e98a59572a4bbb800adc2bd1f3fccc1b

Using Python, extract the Analysis ZIP and every Records ZIP into this
repository root. Members preserve their original extensions/... paths.
Review KAHM_Policy_Augmentation_Archive_Manifest.json before extraction. The source checkout
may already contain identical compact metadata; extract only to a fresh
reproduction checkout and preserve original ZIPs unchanged.

Verify archive files and every member without importing model code
-----------------------------------------------------------------
Run the following from the directory holding the downloaded archives and
the manifest. This uses only the Python standard library, verifies archive
SHA-256, exact membership, every member's SHA-256/size, and each ZIP CRC.

python3 - <<'PY'
import hashlib, json, pathlib, zipfile, zlib
base = pathlib.Path('.')
m = json.loads((base / 'KAHM_Policy_Augmentation_Archive_Manifest.json').read_text())
def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
verified = 0
for a in m['archives']:
    assert pathlib.PurePosixPath(a['file_name']).name == a['file_name']
    path = base / a['file_name']
    assert path.stat().st_size == a['size_bytes']
    assert file_hash(path) == a['sha256']
    expected = {r['path']: r for r in m['members'] if r['archive'] == a['file_name']}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        assert len(infos) == len(expected) == a['member_count']
        assert {i.filename for i in infos} == set(expected)
        for info in infos:
            r = expected[info.filename]
            h, crc, size = hashlib.sha256(), 0, 0
            with z.open(info) as f:
                for b in iter(lambda: f.read(1048576), b''):
                    h.update(b); crc = zlib.crc32(b, crc); size += len(b)
            assert size == info.file_size == r['size_bytes']
            assert h.hexdigest() == r['sha256']
            assert (crc & 0xffffffff) == info.CRC == int(r['crc32'], 16)
            verified += 1
assert verified == m['member_count']
print('Verified archives:', len(m['archives']), 'members:', verified)
PY

Extract the verified members into the fresh checkout
--------------------------------------------------
Run this from the new repository root, replacing ../downloaded-archives with
the directory containing the archive files. Existing identical metadata from
the source checkout is retained. Any conflicting existing file stops restore.

python3 - ../downloaded-archives <<'PY'
import hashlib, json, pathlib, shutil, sys, zipfile
base, root = pathlib.Path(sys.argv[1]), pathlib.Path('.').resolve()
m = json.loads((base / 'KAHM_Policy_Augmentation_Archive_Manifest.json').read_text())
for a in m['archives']:
    assert pathlib.PurePosixPath(a['file_name']).name == a['file_name']
    expected = {r['path']: r for r in m['members'] if r['archive'] == a['file_name']}
    with zipfile.ZipFile(base / a['file_name']) as z:
        for info in z.infolist():
            name = pathlib.PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            target = (root / name).resolve()
            assert target.is_relative_to(root)
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest() == expected[info.filename]['sha256']
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1048576)
print('Restored complete archive into', root)
PY

Recompute the synthetic analysis and audit from saved arrays
-----------------------------------------------------------
Install the declared requirements in an isolated Python environment. From
the restored repository root:

python extensions/policy_augmentation_20261101/analysis_code/analyze_policy.py --output extensions/policy_augmentation_20261101/analysis_recomputed
python extensions/policy_augmentation_20261101/audit/verify_policy.py --analysis extensions/policy_augmentation_20261101/analysis_recomputed --output extensions/policy_augmentation_20261101/verification/policy_audit_recomputed.json

Recompute the natural analysis and audit from saved arrays
---------------------------------------------------------
The frozen natural command intentionally refuses to overwrite outputs.
Preserve the extracted original outputs under different names, then run:

mv extensions/policy_augmentation_natural_20261101/analysis extensions/policy_augmentation_natural_20261101/analysis_saved
mv extensions/policy_augmentation_natural_20261101/audit/verification.json extensions/policy_augmentation_natural_20261101/audit/verification_saved.json
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py analyze
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py verify

This reanalysis does not fit models or create additional independent
observations. A new fit reproduction needs a separate clean output directory;
do not overwrite completed or failed execution records. Per-study README.md
and PROTOCOL.md specify fitting and prediction stages.

Data and inference scope
------------------------
Synthetic: 192 task suites, 24 new independent repetition aggregates, four
generating laws, two noise levels, history 12, horizons 1/4/12. Four fixed
simultaneous primary criteria are evaluated at the repetition level.
Natural: 24 Weekly and 24 Hourly M4 series identities, excluding all 72 previously
studied exact M4 identities. Its last 32 matured origins per series are reserved
within official Train files. The official M4 Test files are NOT downloaded.
The natural panel is descriptive external benchmark-development evidence;
anonymous source series can be dependent. It is not an official M4 ranking.
M4 input commit: 6c1067e5a57161249b17289a565178dc7a3fb3ca
M4 source: https://github.com/Mcompetitions/M4-methods
M4 remains the original data source. Consult its original distribution for
applicable terms; this package does not assign a replacement data license.

Natural P0 deliberately omits the simulation's independent-episode OOF RBF
option and adds causal seasonal-naive and drift controls. Its results are not
pooled with the simulation. Both studies retain the established independently
validation-selected native caps/scalers and fixed calibration roles.

Manifest and integrity scope
----------------------------
KAHM_Policy_Augmentation_Archive_Manifest.json records every member's original relative path,
size, SHA-256, CRC32, archive assignment, and compression. It records archive
hashes/sizes, all complete production-directory inventories, passed audit
identities, and source references. The manifest is external to the ZIP files
to avoid a circular self-hash. It is not itself a member it claims to hash.
Archive parts have at most 400 MiB of original member payload; NPZ and GZ
members remain ZIP_STORED, while text uses DEFLATE level 4. Packaging verifies
all member bytes against their source; it does not recompute model fits.
