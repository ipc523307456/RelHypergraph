import pykeen
import pykeen.datasets as datasets
import pykeen.datasets.analysis as da
import pdb

kg = datasets.FB15k237()
kg.summarize()
print(kg.get_normalized_name())
print(kg.entity_to_id["/m/010016"])
