"""
Run all resource generating scripts
"""

import gen_binding_energies
import gen_element_info

def main():
    gen_binding_energies.main()
    gen_element_info.main()

if __name__ == "__main__":
    main()
