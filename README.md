Ancient Coins Clustering
====


This repository provides classes and functions to conduct a deep learning cluster analysis
on ancient coins stored in a private Database.

## Installation

To run the code, you will need to fill in the configuration file *config.py*
in order to create a connection to the MySQL database.

The following packages are needed to run the script
   * numpy
   * pandas
   * skimage
   * requests
   * mysql.connector
   * PIL
   * pytorch
   * torch



## Issues
   * CoinDataset cannot yet handle coloured images correctly
   * Errors occur when borders are removed on coloured images