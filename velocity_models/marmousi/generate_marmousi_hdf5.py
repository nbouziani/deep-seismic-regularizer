import os

from SeismicMesh import write_velocity_model

path = ''
name = 'marmvel.segy'
filename = os.path.join(path, name)

name = 'marmvel'
ofname = os.path.join(path, name)

write_velocity_model(filename, ofname)