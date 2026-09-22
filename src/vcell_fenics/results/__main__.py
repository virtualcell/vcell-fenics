"""``python -m vcell_fenics.results BUNDLE [--require-status S] [--json]`` — summarise a results bundle."""

import sys

from vcell_fenics.results.reader import main

sys.exit(main())
