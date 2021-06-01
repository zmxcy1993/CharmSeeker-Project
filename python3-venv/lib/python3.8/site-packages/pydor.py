# Copyright (c) Jeremías Casteglione <jrmsdev@gmail.com>
# See LICENSE file.

import sys

from configparser import ConfigParser, ExtendedInterpolation

__all__ = ['main']

# config manager

class _Config(ConfigParser):

	def __init__(self):
		super().__init__(defaults = {},
			allow_no_value = False,
			delimiters = ('=',),
			comment_prefixes = ('#',),
			strict = True,
			interpolation = ExtendedInterpolation(),
			default_section = 'default')

# helper objects

config = _Config()

# main

def main():
	return 0

if __name__ == '__main__':
	sys.exit(main())
