import ray
import nums
import nums.numpy as nps


# Initialize ray and connect it to the cluster.
ray.init(address="auto")
# Initialize nums with the cluster shape. Here we set it to use all the nodes in the ray cluster.
nums.init(cluster_shape=(len(ray.nodes()), 1))


def main():
    X = nps.random.rand(10**4)
    Y = nps.random.rand(10**4)
    Z = nps.add(X, Y)
    print("X + Y = ", Z.get())


if __name__ == "__main__":
    main()
