import ray
import nums
import nums.numpy as nps
from nums.core import settings


# Initialize ray and connect it to the cluster.
ray.init(address="auto")
# Set the cluster shape for nums. Here we set it to use all the nodes in the ray cluster. 
settings.cluster_shape = (len(ray.nodes())-1, 1)


def main():
    X = nps.random.rand(10**4)
    Y = nps.random.rand(10**4)
    Z = nps.add(X,Y)
    print("X + Y = ",Z.get())


if __name__ == "__main__":
    main()
