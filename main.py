import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) # assets are loaded by relative path

from Core import Core

if __name__ == '__main__':
    oCore=Core()
    oCore.main_loop()
