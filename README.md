# GSLS Quantification

Contains source-code for experimentation with PCC, EM, and GSLS
quantification methods, as used in the paper "Dynamic Quantification
with Constrained Error Under Unknown General Dataset Shift".

## Dependencies

In order to run this project, you must have the following dependencies
installed on your host:

* [Docker Community Edition](https://docs.docker.com/get-docker/) (>= 17.09)
* [Docker Compose](https://docs.docker.com/compose/install/) (>= 1.17)
  (Included with Docker Desktop on Mac/Windows)
* [Make](https://www.gnu.org/software/make/) (technically optional if
  you don't mind running the commands in the Makefile directly)

**Note:** If you use [Git bash](https://git-scm.com/downloads) on
Windows and also
[install `make`](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058)
into Git bash, then you should be able to run this project on Windows.

## Reproducing Results

1. Ensure the dependencies listed above are installed.
2. Run `make run` in this directory.
   * This will perform all Docker image build steps and dependency
     installations every time you run it, so that you can never forget
     to rebuild. The first time you run this, it make take some time
     for the base Docker image and other dependencies to be
     downloaded.
3. Browse to http://localhost:9999 and enter the token displayed in
   the terminal (or just follow the link in the terminal).
4. Run the following Jupyter notebooks in the order listed here to
   reproduce the results:

| Notebook | Description |
| -------- | ----------- |
| [01_ExampleGslsHistograms](http://localhost:9999/notebooks/01_ExampleGslsHistograms.ipynb) | Histogram plots used to demonstrate GSLS model. |
| [02_QuantificationSensitivityAnalysis](http://localhost:9999/notebooks/02_QuantificationSensitivityAnalysis.ipynb) | Analysis of GSLS quantification's sensitivity to histogram bin and target instance counts. |
| [03_QuantificationUnderGslsShift](http://localhost:9999/notebooks/03_QuantificationUnderGslsShift.ipynb) | Comparison of quantification methods under GSLS shift. |
| [04_QuantificationUnderPriorShift](http://localhost:9999/notebooks/04_QuantificationUnderPriorShift.ipynb) | Comparison of quantification methods under prior shift. |
| [05_QuantificationPaperTables](http://localhost:9999/notebooks/05_QuantificationPaperTables.ipynb) | Generation of tables of quantification results presented in the paper from cached experiment outcomes. |
| [06_ShiftTestsRuntime](http://localhost:9999/notebooks/06_ShiftTestsRuntime.ipynb) | Comparison of the runtime of shift tests. |
| [07_ShiftTestsUnderGslsShift](http://localhost:9999/notebooks/07_ShiftTestsUnderGslsShift.ipynb) | Comparison of shift tests under GSLS shift. |
| [08_ShiftTestsUnderPriorShift](http://localhost:9999/notebooks/08_ShiftTestsUnderPriorShift.ipynb) | Comparison of shift tests under prior shift. |
| [09_ShiftTestTables](http://localhost:9999/notebooks/09_ShiftTestTables.ipynb) | Generation of tables of shift test results presented in the paper from cached experiment outcomes. |
| [10_QuantificationAndShiftTestsUnderSampleShift](http://localhost:9999/notebooks/10_QuantificationAndShiftTestsUnderSampleShift.ipynb) | Comparison of quantification methods and shift tests under sample (real-world) shift. |
| [11_SampleShiftQuantificationAndShiftTestTables](http://localhost:9999/notebooks/11_SampleShiftQuantificationAndShiftTestTables.ipynb) | Generation of tables of quantification and shift test results for sample (real-world) shift presented in the paper from cached experiment outcomes. |
| [12_RejectionUnderSampleShiftRuntime](http://localhost:9999/notebooks/12_RejectionUnderSampleShiftRuntime.ipynb) | Comparison of the runtime of rejection (constrained quantification) methods. |
| [13_RejectionUnderSampleShift](http://localhost:9999/notebooks/13_RejectionUnderSampleShift.ipynb) | Comparison of rejection (constrained quantification) methods under sample (real-world) shift. |
| [14_RejectionUnderSyntheticShift](http://localhost:9999/notebooks/14_RejectionUnderSyntheticShift.ipynb) | Comparison of rejection (constrained quantification) methods under synthetic (GSLS and prior) shift. |
| [15_RejectionTables](http://localhost:9999/notebooks/15_RejectionTables.ipynb) | Generation of tables of rejection (constrained quantification) results presented in the paper from cached experiment outcomes. |
| [16_SyntheticExample](http://localhost:9999/notebooks/16_) | Quantification tests with simple synthetic datasets. |
| [17_BootstrapQuantificationUnderGslsShift](http://localhost:9999/notebooks/17_BootstrapQuantificationUnderGslsShift.ipynb) | Comparison of bootstrap quantification methods under GSLS shift. |
| [18_BootstrapQuantificationUnderPriorShift](http://localhost:9999/notebooks/18_BootstrapQuantificationUnderPriorShift.ipynb) | Comparison of bootstrap quantification methods under prior shift. |
| [19_BootstrapQuantificationUnderSampleShift](http://localhost:9999/notebooks/19_BootstrapQuantificationUnderSampleShift.ipynb) | Comparison of bootstrap quantification methods under sample (real-world) shift. |

### Linting

You can run [flake8](http://flake8.pycqa.org/en/latest/) linting on
your modules with: `make lint`.

### Testing

You can run [pytest](https://docs.pytest.org/en/latest/) unit tests
linting contained in your modules with: `make test`.

An HTML code-coverage reported will be generated for each module at:
`<module-dir>/test/coverage/index.html`.

## Managing Data

Large data files are cached in the `data/` directory so that they are
not committed to your Git repository and not transmitted to the Docker
daemon during image builds (see [.dockerignore](#dockerignore)).

You may not want to commit the outputs of notebook cells to the Git
repository. If you have Python 3 installed, you can use
[nbstripout](https://github.com/kynan/nbstripout) to configure your
Git repository to exclude the outputs of notebook cells when running
`git add`:

1. `python3 -m pip install nbstripout nbconvert`
2. Run `nbstripout --install` in this directory (installs hooks into
   `.git`).

## Notes About Docker

### Opening a Shell

If you would like to open a bash shell inside the Docker container
running the Jupyter notebook server, use: `make bash` or `make
sudo-bash`. If `make run` is not currently running, you can instead
use `make run-bash` or `make run-sudo-bash`.

### System Dependencies and Other OS Configurations

To install system packages or otherwise alter the Docker image's
operating system, you can make changes in the Dockerfile. An example
section that will allow you to install `apt` packages is included.

### Changing the Jupyter Server Port

Change the `ports` entry in `docker-compose.yml` to:
`'YOURPORT:9999`, then re-run `make run`.

### .dockerignore

Whenever the Docker image is rebuilt (after certain files are
changed), Docker will transmit the contents of this directory to the
Docker daemon.

To speed up build times, you should add an entry to the
`.dockerignore` file for any directories containing large files you do
not need to be included in the Docker image at build time.

### Managing Docker Image Storage

When Docker builds new versions of its images, it does not delete the
old versions. Over time, this can lead to a significant amount of disk
space being used for old versions of images that are no longer needed.

Because of this, you may wish to periodically run `docker image prune`
to delete any "dangling images" - images that aren't currently
associated with any image tags/names.

## Using Jupyter from Emacs

Did you know you can work with Jupyter notebooks from Emacs? All you
need to do is install
[EIN](http://millejoh.github.io/emacs-ipython-notebook/): `M-x
package-refresh-contents <Enter> M-x package-install <Enter> ein`

### Connecting EIN to your Jupyter server

1. Ensure `make run` is running.
2. `M-x ein:login` (URL: http://127.0.0.1:9999, Password: token from `make run`)
3. `M-x ein:notebooklist-open`

### Common EIN Commands

```
M-<enter> - Execute cell and move to next.
C-c C-c - Execute cell.
C-c C-z - Interrupt command
C-c C-x C-r - Restart session
C-<up/down> - Navigate cells.
M-<up/down> - Move cells.
C-c C-b - Insert cell below (C-a for above).
C-c C-l - Clear cell output.
C-c C-k - Delete cell.
C-c C-f - Open file.
C-c C-h - Help at cursor.
C-c C-S-l - Clear all output.
C-c C-t - Toggle cell type.
```
