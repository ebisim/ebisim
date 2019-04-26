"""
Run all resource generating scripts
"""

import binding_energies
import element_info

def main():
    binding_energies.main()
    element_info.main()

if __name__ == "__main__":
    main()
