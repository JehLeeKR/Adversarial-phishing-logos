import os
from gap_util import str_project_root_path

def get_all_brands(folder_path, output_path):
    count = 0
    with open(output_path, mode='w+') as out:
        for dir_name in os.listdir(folder_path):
            if not dir_name.startswith('.'):
                count += 1
                out.writelines(dir_name + '\n')
    print(f'Total brands count: {count}')


def main():
    get_all_brands(f"{str_project_root_path}/data/datasets_logo_181/test", 
                   f"{str_project_root_path}/data/datasets_logo_181/test/classes.txt")

if __name__ == '__main__':
    main()
