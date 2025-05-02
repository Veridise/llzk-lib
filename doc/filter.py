"""
This file is used by Doxygen to filter input files. We use this filtering
capability to remove HTML tags from section headers for the Table of Contents,
as they are not properly rendered there.
Fortunately, currently, <tt> is only used in the ToC and is
"""

import sys
import re

for line in sys.stdin:
    line = re.sub(r'<tt>(.*?)</tt>', r'\1', line)
    sys.stdout.write(line)