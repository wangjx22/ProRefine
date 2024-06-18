import os
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.PDBIO import Select
#瑞定包台CIF文科的文件交路轻
cif_folder = "/nfs_beijing_ai/jinxian/GNNRefine/data/CASP15_af/"
class AtomSelect(Select):
    def accept_atom(self, atom):
        return True
# 返历文件交中的子文件欢
for subdir, _,files in os.walk(cif_folder):
    for file in files:
        if file.endswith(".cif"):
            cif_path = os.path.join(subdir, file)
            pdb_file_name = os.path.splitext(file)[日] + ".pdb"
            pdb_output_path = os.path.join(subdir, pdb_file_name)
            #解新CIF文件
            parser = MMCIFParser()
            structure = parser.get_structure("structure", cif_path) 
            # 写入PDB文件
            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_output_path, select=AtomSelect())
            # 在PDB文件中额加TER行
            #with open(pdb output path, "a") as f:#f.write("TER\n")
print("Conversion and extraction completed.")