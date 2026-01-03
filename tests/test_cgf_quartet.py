import unittest
import numpy as np
import os
import json

from ppmil import GTO, CGF
from ppmil import IntegralEvaluator, HuzinagaElectronRepulsionEngine, HellsingElectronRepulsionEngine

class TestRepulsion(unittest.TestCase):

    def test_reference_set_00(self):
        run_eri_reference_tests(os.path.join(os.path.dirname(__file__), 'data', 'refsets', '001_base.json'), 1e-8, 5e-7)
    
    def test_reference_set_01(self):
        run_eri_reference_tests(os.path.join(os.path.dirname(__file__), 'data', 'refsets', '002_h2.json'), 1e-8, 5e-7)

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
    json_filename,
    rtol=1e-12,
    atol=1e-14
):
    """
    Load ERI reference JSON file and execute all tests.
    """
    with open(json_filename, "r") as f:
        data = json.load(f)

    normalization_mode = data["conventions"]["normalization"]

    integrator = IntegralEvaluator(
        None, None, HuzinagaElectronRepulsionEngine()
    )

    for test in data["tests"]:
        run_single_eri_test(
            test,
            integrator,
            normalization_mode,
            rtol,
            atol
        )

    print(f"All {len(data['tests'])} ERI reference tests passed.")

if __name__ == '__main__':
    unittest.main()