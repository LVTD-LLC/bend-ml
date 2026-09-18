# bend-ml: machine learning in Bend 2. Parallel linear regression, batches,
# statistics, and metrics. https://github.com/LVTD-LLC/bend-ml
#
# Package entry point: publish with `bend bend_ml.bend --publish`.
# Import individual APIs from <package-hash>/src/batch.bend,
# src/stats.bend, src/linear_regression.bend, src/metrics.bend, and
# src/numeric.bend. This entry also checks the library's structural proofs.

import ./src/batch.bend as Batch
import ./src/numeric.bend as Numeric
import ./src/stats.bend as Stats
import ./src/metrics.bend as Metrics
import ./src/linear_regression.bend as Linear
import ./PROOF.bend as Proof

# The publisher bundles imported source files only, so the distribution's
# license is included here as well as in the repository's LICENSE file.
# MIT License
#
# Copyright (c) 2026 LVTD LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
