import unittest
import numpy as np
import os
import json

from ppmil import GTO, CGF
from ppmil import IntegralEvaluator, HuzinagaElectronRepulsionEngine, HellsingElectronRepulsionEngine

ENGINES = [
    HuzinagaElectronRepulsionEngine,
    HellsingElectronRepulsionEngine
]

class TestRepulsion(unittest.TestCase):

    def test_reference_set_00(self):
        for e in ENGINES:
            run_eri_reference_tests(e, os.path.join(os.path.dirname(__file__), 'data', 'refsets', '001_base.json'), 1e-8, 5e-7)
    
    def test_reference_set_01(self):
        for e in ENGINES:
            run_eri_reference_tests(e, os.path.join(os.path.dirname(__file__), 'data', 'refsets', '002_h2.json'), 1e-8, 5e-8)

    def test_reference_set_02(self):
        for e in ENGINES:
            run_eri_reference_tests(e, os.path.join(os.path.dirname(__file__), 'data', 'refsets', '003_ham.json'), 1e-14, 1e-14)

def make_gto_from_json(gto_json, normalization_mode):
    """
    Construct a GTO from JSON specification.
    """
    alpha = gto_json["exponent"]
    coeff = gto_json["coefficient"]
    center = gto_json["center"]
    angmom = gto_json["angular_momentum"]

    gto = GTO(coeff, alpha, center, angmom)

    if normalization_mode == "raw_primitive":
        # cancel primitive normalization
        gto.c /= gto.norm
    elif normalization_mode in (
        "normalized_primitive",
            "normalized_contracted",
    ):
        # default behavior: nothing to do
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {normalization_mode}")

    return gto


def make_cgf_from_json(cgf_json, normalization_mode):
    """
    Construct a CGF from JSON specification.
    """
    cgf = CGF()
    for gto_json in cgf_json["gtos"]:
        gto = make_gto_from_json(gto_json, normalization_mode)
        cgf.gtos.append(gto)
    return cgf

def run_single_eri_test(test_json, integrator, normalization_mode, rtol, atol):
    """
    Run a single ERI test from JSON.
    """
    cgfs = [
        make_cgf_from_json(cgf_json, normalization_mode)
        for cgf_json in test_json["cgfs"]
    ]

    computed = integrator.repulsion(cgfs[0], cgfs[1], cgfs[2], cgfs[3])
    expected = test_json["eri"]["value"]

    np.testing.assert_allclose(
        computed,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"ERI test failed: {test_json.get('id', '<unnamed>')}"
    )

def run_eri_reference_tests(
    engine,
    json_filename,
    rtol=1e-12,
    atol=1e-14,
    stop_on_fail=True,
    verbose=True,
):
    """
    Load ERI reference JSON file and execute all tests
    with detailed per-test diagnostics.
    """
    with open(json_filename, "r") as f:
        data = json.load(f)

    normalization_mode = data["conventions"]["normalization"]

    integrator = IntegralEvaluator(
        None, None, engine()
    )

    ntests = len(data["tests"])
    nfail = 0

    if verbose:
        print("=" * 72)
        print(f"ERI reference test suite: {json_filename}")
        print(f"Normalization mode: {normalization_mode}")
        print(f"rtol = {rtol:.1e}, atol = {atol:.1e}")
        print("=" * 72)

    for i, test in enumerate(data["tests"], start=1):
        test_id = test.get("id", f"<unnamed-{i}>")
        desc = test.get("description", "")

        cgfs = [
            make_cgf_from_json(cgf_json, normalization_mode)
            for cgf_json in test["cgfs"]
        ]

        computed = integrator.repulsion(
            cgfs[0], cgfs[1], cgfs[2], cgfs[3]
        )
        expected = test["eri"]["value"]

        abs_err = abs(computed - expected)
        rel_err = abs_err / abs(expected) if expected != 0.0 else math.inf

        passed = (
            abs_err <= atol or
            rel_err <= rtol
        )

        if verbose:
            print(f"[{i:03d}/{ntests:03d}] {test_id}")
            if desc:
                print(f"  {desc}")
            print(f"  expected = {expected:.17e}")
            print(f"  computed = {computed:.17e}")
            print(f"  abs_err  = {abs_err:.3e}")
            print(f"  rel_err  = {rel_err:.3e}")
            print(f"  result   = {'PASS' if passed else 'FAIL'}")
            print("-" * 72)

        if not passed:
            nfail += 1
            if stop_on_fail:
                raise AssertionError(
                    f"ERI test failed: {test_id}\n"
                    f"expected = {expected:.17e}\n"
                    f"computed = {computed:.17e}\n"
                    f"abs_err  = {abs_err:.3e}\n"
                    f"rel_err  = {rel_err:.3e}"
                )

    if nfail == 0:
        print(f"All {ntests} ERI reference tests PASSED.")
    else:
        raise AssertionError(
            f"{nfail} / {ntests} ERI reference tests FAILED."
        )

if __name__ == '__main__':
    unittest.main()