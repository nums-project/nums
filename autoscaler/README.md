These instructions explain how to set up a Ray cluster with NumS on AWS. 
They supplement the well commented ```autoscaler-aws.yaml``` file.
Refer to the [ray autoscaler](https://docs.ray.io/en/master/cluster/cloud.html) page for additional details. 

Prior to using this autoscaler, nums and boto3 must be installed on your local machine (```pip install nums boto3```) with your AWS credentials configured, as described in the [boto docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html). If you use conda, make sure to activate the correct conda environment.

# Steps for successfully launching a NumS cluster on AWS:
This doc aims to provide a simple cluster configuration file to launch a NumS cluster on AWS.
To get started, follow these steps in order.

[A. Configuration](#a-configuration) \
[B. Running the configuration file](#b-running-the-configuration-file) \
[C. Example](#c-example) \
[D. Destroying the cluster](#d-destroying-the-cluster)

 \
The below walkthrough provides the following:
  * Steps to set the configuration options for the ```autoscaler-aws.yaml``` file.
  * Steps to launch the cluster using this configuration file.
  * Provide a simple example to run on the cluster.
  * Steps to destroy the cluster.

## A. Configuration
In the ```autoscaler-aws.yaml``` file: 
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
This will configure the head node. 
You can choose the ec2 instance type, disk and AMI for the head node here. 
```
available_node_types:
    nums.head.default:
    .
    .
    node_config:
            InstanceType: r5.16xlarge
            ImageId: ami-08c6f8e3871c56139 # Deep Learning AMI (Ubuntu) Version 46
            BlockDeviceMappings:
                - DeviceName: /dev/sda1
                  Ebs:
                      VolumeSize: 100
```
Similarly, for the ```nums.worker.default``` field, edit the ```min_workers``` key 
to set the number of NumS workers.
Set the ```max_workers``` key to the same value.
```
	nums.worker.default:
        	min_workers: 4
        	max_workers: 4
```
Edit worker node configurations here under ```node_config``` field in a similar way as done before for head node above.
Make sure to use the correct AMI as per your region and availability zones.
4. Modify the ```file_mounts``` field to indicate any directories or files to copy from the local machine to every node on the cluster, or leave it commented.
```
file_mounts: {
#    "/path2/on/remote/machine": "/path2/on/local/machine",
}
```

## B. Running the configuration file

5. To launch a NumS cluster, run 
```
ray up autoscaler-aws.yaml -y
```
* This will launch a ray cluster with NumS. 

## C. Example
After you launch the cluster using the steps above, you can refer to [this example](https://github.com/nums-project/nums/blob/main/autoscaler/example.py).
* In this example:
  * We first initialize ray and set the nums cluster shape.
  ```
  # Initialize ray and connect it to the cluster
  ray.init(address="auto")
  # Set the cluster shape for nums. Here we set it to use all the nodes in the ray cluster.
  settings.cluster_shape = (len(ray.nodes())-1, 1)
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
  cd <local_path_to_this_repo>/autoscaler
  ray rsync-up autoscaler-aws.yaml 'example.py' '/home/ubuntu'
  ```
  * Then ssh on the head node of the cluster 
  ```
  ray attach autoscaler-aws.yaml
  ```
  * Then on the head node, run ```python example.py``` to run this example on the cluster.
  * Then ```exit``` to terminate the ssh connection to the head node. 


## D. Destroying the cluster
7. To destroy the cluster, run the following from your local machine.
```
ray down autoscaler-aws.yaml
```
* Tip: In ```autoscler-aws.yaml``` file, set ```cache_stopped_nodes``` key  to ```False``` in the ```provider``` field to terminate the nodes and disable reuse, instead of shutting them down upon running step 7.
