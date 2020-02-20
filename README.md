Ancient Coins Clustering
====


This repository provides classes and functions to conduct deep learning analyses
on ancient coins stored in a private Database.

## Installation

When pulling this repository, you will need to fill in the configuration file *config.py*
in order to be able to create a connection to the MySQL database.

The following packages will be needed for running the script
   * numpy
   * pandas
   * skimage
   * requests
   * mysql.connector
   * PIL
   * pytorch
   * torch



## Issues
   * CoinDataset cannot yet handle coloured images
   * Errors occur when borders are removed on coloured images