import os
from setuptools import setup

# Force platform specific wheel
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    # https://stackoverflow.com/a/45150383/1255535
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    print('Warning: cannot import "wheel" package to build platform-specific wheel')
    print('Install the "wheel" package to fix this warning')
    bdist_wheel = None

cmdclass = {"bdist_wheel": bdist_wheel} if bdist_wheel is not None else dict()


# Read requirements.txt
with open("requirements.txt", "r") as f:
    lines = f.readlines()
install_requires = [line.strip() for line in lines if line]


def find_stubs(package):
    stubs = []
    for root, _, files in os.walk(package):
        for file in files:
            path = os.path.join(root, file).replace(package + os.sep, "", 1)
            stubs.append(path)
    return {package: stubs}


setup(
    author="Manh Ha Hoang",
    author_email="manhha1402@gmail.com",
    description=["tensorrt inferece: Nvidia tensorrt inference Library "],
    cmdclass=cmdclass,
    install_requires=install_requires,
    include_package_data=True,
    license="MIT",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # Name of the package on PyPI
    name="@PYPI_PACKAGE_NAME@",
    packages=[
        "tensorrt_inference",
    ],
    packages_data=find_stubs("tensorrt_inference-stubs"),
    # url="@PROJECT_HOME@",
    # project_urls={
    #     "Documentation": "@PROJECT_DOCS@",
    #     "Source code": "@PROJECT_CODE@",
    #     "Issues": "@PROJECT_ISSUES@",
    # },
    version="@PROJECT_VERSION@",
    zip_safe=False,
)
