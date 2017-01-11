from setuptools import setup, find_packages

setup(
    name="dihedral_model",
    version="1.0.1",
    author="Brandon Wood",
    author_email="b.wood@berkeley.edu",
    install_requires=["pymatgen>=4.5.4"],
    extras_require={"babel": ["openbabel", "pybel"]},
    packages=find_packages(),

)
