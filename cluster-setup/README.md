These instructions explain how to set up a Ray cluster with NumS on AWS. 
They supplement the well commented ```aws-cluster.yaml``` file.
Refer to the [ray cluster setup](https://docs.ray.io/en/master/cluster/cloud.html) page for additional details. 

Prior to using this script, nums and boto3 must be installed on your local machine (```pip install nums boto3```) with your AWS credentials configured, as described in the [boto docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html). If you use conda, make sure to activate the correct conda environment.

# Steps for successfully launching a NumS cluster on AWS:
This doc aims to provide a simple cluster configuration file to launch a NumS cluster on AWS.
To get started, follow these steps in order.

[A. Configuration](#a-configuration) \
[B. Running the configuration file](#b-running-the-configuration-file) \
[C. Example](#c-example) \
[D. Destroying the cluster](#d-destroying-the-cluster)

 \
The below walkthrough provides the following:
  * Steps to set the configuration options for the ```aws-cluster.yaml``` file.
  * Steps to launch the cluster using this configuration file.
  * Provide a simple example to run on the cluster.
  * Steps to destroy the cluster.

## A. Configuration
In the ```aws-cluster.yaml``` file: 
1. Modify the ```max_workers``` key to set the global max workers that may launch in addition to the head node.
```
# The maximum number of workers nodes to launch in addition to the head node.
max_workers: 100
```
2. Modify the parameters in the ```provider``` field for your aws specific configurations regarding regions and availability zones.
```
# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    availability_zone: us-west-2a,us-west-2b
```
3. In the ```available_node_types``` field, edit the ```node_config``` field for ```nums.head.default```. 
This will configure the head node. You can choose the ec2 instance type, disk and AMI for the head node here. 
Also edit the resources field to set the number of CPUs nums would use. A good rule of thumb is to set it to number of physical cores - 2. 
For example for r5.16xlarge machines, we set it to 32 - 2 = 30. 
```
available_node_types:
    nums.head.default:
    ..
    resources: {"CPU": 30}
    ..
    node_config:
            InstanceType: r5.16xlarge
            ImageId: ami-0050625d58fa27b6d # Deep Learning AMI (Ubuntu 18.04) Version 50
            BlockDeviceMappings:
                - DeviceName: /dev/sda1
                  Ebs:
                      VolumeSize: 120
```
Similarly, for the ```nums.worker.default``` field, edit the ```min_workers``` key 
to set the number of NumS workers.
Set the ```max_workers``` key to the same value.
```
	nums.worker.default:
        	min_workers: 3
        	max_workers: 3
```
Edit worker node configurations here under the ```node_config``` field and number of CPUs under `resources` field in a similar way as done before for the head node above.
Make sure to use the correct AMI as per your region and availability zones.

4. Modify the ```file_mounts``` field to indicate any directories or files to copy from the local machine to every node on the cluster, or leave it commented.
```
file_mounts: {
#    "/path1/on/remote/machine": "/path1/on/local/machine",
#    "/path2/on/remote/machine": "/path2/on/local/machine",
}
```

## B. Running the configuration file

5. To launch a NumS cluster, run 
```
ray up aws-cluster.yaml
```
* This will launch a ray cluster with NumS. 

## C. Example
After you launch the cluster using the steps above, you can refer to [this example](https://github.com/nums-project/nums/blob/master/cluster-setup/example.py).
* In this example:
  * We first initialize ray and nums.
  ```
  # Initialize ray and connect it to the cluster.
  ray.init(address="auto")
  # Initialize nums with the cluster shape. Here we set it to use all the nodes in the ray cluster.
  nums.init(cluster_shape=(len(ray.nodes()), 1))
  ```
  * Then we create two nums arrays with random values that get created in a distributed fashion on the worker nodes of the cluster. 
  ```
  def main():
    X = nps.random.rand(10**4)
    Y = nps.random.rand(10**4)
  ```
  * We then perform a sum operation on the two nums arrays.
  ```
    Z = nps.add(X,Y)
  ```
  * Then we do ```Z.get()``` which waits and fetches final values of ```X+Y```.
  ```
    print("X + Y = ", Z.get())
  ```

6. To run this example:
  * First run the following commands from your local machine to copy the example.py file to head node of the cluster
  ```
  cd <local_path_to_this_repo>/cluster-setup
  ray rsync-up aws-cluster.yaml 'example.py' '/home/ubuntu'
  ```
  * Then ssh on the head node of the cluster 
  ```
  ray attach aws-cluster.yaml
  ```
  * Then on the head node, run ```python example.py``` to run this example on the cluster.
  * Then ```exit``` to terminate the ssh connection to the head node. 


## D. Destroying the cluster
7. To destroy the cluster, run the following from your local machine.
```
ray down aws-cluster.yaml
```
* Tip: In ```aws-cluster.yaml``` file, set ```cache_stopped_nodes``` key  to ```False``` in the ```provider``` field to terminate the nodes and disable reuse, instead of shutting them down upon running step 7.
