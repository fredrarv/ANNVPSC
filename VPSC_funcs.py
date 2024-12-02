import shutil
import subprocess
import os

def compileVPSC(bat_file):
    try:
        # Compile the vpsc8.exe from the .bat file and capture the output
        result = subprocess.run([bat_file],shell=True,check=True,capture_output=True,text=True)
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        # In case of an error
        print(f"Error occurred: {e}")
        print("Error output:\n", e.stderr)

    def make_texture_list(num):

    """Function to make a list of paths to textures for input into yield surface calculator"""
    texturesEveryTen = pd.read_csv(texturePath,sep='\s+',skiprows=skipIndices,names=['a','b','c','d'])

    texturesEveryTenFiltered = texturesEveryTen.to_numpy().reshape(-1,1000,4)
    
    textureFiles  = []
    outputFileNames  = []

    for i in range(texturesEveryTenFiltered.shape[0]):
            
        sub_array = texturesEveryTenFiltered[i]
        df = pd.DataFrame(sub_array) 

        filename = f'texture_{num}_{i+1}.tex'
        newFileNamePCYS = f'PCYS_{num}_{i+1}.OUT'

        file_path = os.path.join(VPSC_directory, filename)
        
        textureFiles.append((file_path,f'textures_{num}\{filename}'))
        outputFileNames.append(newFileNamePCYS)
    
    return textureFiles,outputFileNames