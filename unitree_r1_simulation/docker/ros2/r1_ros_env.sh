source /opt/ros/humble/setup.bash
source /opt/unitree_ws/install/setup.bash
source /opt/r1_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${DDS_DOMAIN_ID:-1}"
cat > /tmp/cyclonedds_r1.xml <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="${DDS_INTERFACE:-lo}" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>
EOF
export CYCLONEDDS_URI=file:///tmp/cyclonedds_r1.xml
