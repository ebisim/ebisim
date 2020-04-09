"""
Run all resource generating scripts
"""

import gen_shell_data
import gen_element_data

def main():
    gen_shell_data.main()
    gen_element_data.main()

if __name__ == "__main__":
    main()
