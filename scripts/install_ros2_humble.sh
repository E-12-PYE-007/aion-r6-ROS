#!/usr/bin/env bash
# Installs ROS 2 Humble (Debian packages) on Ubuntu 22.04 (jammy).
# Intended for the Jetson companion computer. Run as the target user with sudo access.
set -euo pipefail

ROS_DISTRO=humble
ROS_PACKAGE=ros-humble-ros-base  # Headless variant; no RViz/GUI tools.

CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
if [ "$CODENAME" != "jammy" ]; then
    echo "This script targets Ubuntu 22.04 (jammy); detected '$CODENAME'." >&2
    exit 1
fi

# Locale must be UTF-8 for ROS 2.
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Ubuntu Universe repo is required for some ROS 2 dependencies.
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

# Register the ROS 2 apt repository via the official ros2-apt-source package.
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb"
sudo apt install -y /tmp/ros2-apt-source.deb
rm -f /tmp/ros2-apt-source.deb

# Install ROS 2 and build tooling.
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
    "$ROS_PACKAGE" \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-argcomplete

# rosdep init is a one-time, machine-wide step.
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# Source ROS 2 in every new shell.
if ! grep -qxF "source /opt/ros/${ROS_DISTRO}/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
fi

echo "ROS 2 ${ROS_DISTRO} installed. Run 'source ~/.bashrc' or start a new shell to pick it up."
