import numpy as np
import pandas as pd
import requests
import os
import warnings
import config as cf
import mysql.connector as sql
from PIL import Image
from io import BytesIO
from skimage import io, transform

"""
### EXAMPLE CODE EXECUTION TO CREATE COIN DATASET ###


conn = sql.connect(**cf.config)

coin_query = CoinQuery(conn, side='front', coin_type='original', n_max=np.inf)
coin_query.create_dataset()

conn.close()
"""

base_url = "https://digilib.bbaw.de/digitallibrary/servlet/Scaler?fn=silo10/thrakien/"
warnings.filterwarnings('error')

class ImageParams():
    """
    Helper class that defines the formatting of the images
    when saved in the folder structure

    Attributes:
        size (int): Size of the 2d image, resulting in an image of dim=(size,size)
        cut_edges (bool): Boolean to determine whether images should be cropped up to the coin borders
        as_gray (bool): Determines whether the image should be coloured
    """
    # IDEA: Maybe add some additional formatting (especially for plastercasts)
    # ToDo: Implement edge cutting for RGB images 
    def __init__(self, size=128, cut_edges=True, as_gray=True):
        self.size = size
        self.cut_edges = cut_edges
        self.as_gray = as_gray

class CoinQuery():
    """
    Coin database query creation class.
    Instantiation results in a dictionary of coins with classes as keys.
    Dataset can be created subsequently.

    Attributes:
        conn (Connection): Database connection object
        __params (Dict):  Database query parameters
        image_params (ImageParams): Formatting parameters of coin images
    """

    def __init__(self, conn, side='front', coin_type='original', n_min=20, n_max=100):
        """
        Query and coin dictionary is created on object instantiation according to given arguments.
        
        Args:
            conn (Connector): Database connection object

        Optional Args:
            side (str):       Side(s) of the coins to be retrieved from database
            coin_type (str):  Type(s) of coins to be retrieved from database
            n_min (int):      Minimum amount of coins per class for retrieval
            n_max (int):      Maximum amount of coins per class for retrieval
        """
        self.conn = conn
        self.__params = {'side':side, 'coin type':coin_type, 'Minimum of coins per class':n_min, 'Maximum of coins per class':n_max}
        self.image_params = ImageParams()

        if side not in ('front', 'back', 'both'):
            raise ValueError('Attribute \'side\' needs to be \'front\', \'back\' or \'both\'')
        if coin_type not in ('original', 'plastercast', 'both'):
            raise ValueError('Attribute \'side\' needs to be \'original\', \'plastercast\' or \'both\'')
        if n_min < 0:
            raise ValueError('Attribute \'n_min\' needs to be positive')
        if n_max < 0:
            raise ValueError('Attribute \'n_max\' needs to be positive')        
        
        classes = []
        if side is not 'both':
            classes = self.__classes_query(side)

        self.coins = self.__query(side, coin_type, n_min, n_max, classes)

    
    def __classes_query(self, side):
        """
        If only single coin side if selected for processing, new classes need to be inferred.
        This is done according to Design, Legend, ControlMark and LegendDirection.
        Original classes are aggregated correspondingly and retrieved from the database.
        """
        classes = []
        have_count = "having cnt > 5"
        s = 'O'
        if side is 'back':
            s = 'R'       
        
        query = """
                SELECT count(*) AS cnt, group_concat(TypeID)
                FROM types
                GROUP BY Design_{0}, Legend_{0},
                ControlMark_{0}, LegendDirection_{0}
                {1}
                """.format(s, have_count)

        cursor = self.conn.cursor()
        cursor.execute(query)
        
        for row in cursor:
            classes.append(row[1])

        return classes

    def __execute_query(self, query, coin_class=None):
        coins = []
        if coin_class is not None:
            query = query.format(tuple(coin_class.split(',')))

        cursor = self.conn.cursor()
        cursor.execute(query)

        for row in cursor:
            coins.append(row)

        return coins


    def __query(self, side, coin_type, n_min, n_max, classes):
        """
        Retrieve the coin data per class from the database according to given arguments
        """
        coins = {}        
        query = """
                SELECT images.CoinID, type_coin_helper.TypeID,
                ObverseImageFilename, images.ObjectType, images.Path
                FROM images
                INNER JOIN type_coin_helper ON
                images.CoinID = type_coin_helper.CoinID                
                """

        if coin_type is not 'both':
                query += "WHERE images.ObjectType = '{0}'".format(coin_type)

        # Only single side of coins is processed
        # Aggregated classes are used
        if classes:
            query += " AND type_coin_helper.TypeID IN {}"
            i = 0
            for c in classes:              
                res = self.__execute_query(query, c)
                if len(res) < n_min:
                    continue
                if len(res) > n_max:
                    res[:n_max]
                coins[i] = res
                i += 1
        # Both sides of coins are processed
        # Original classes are used    
        else:
            res = self.__execute_query(query)
            idx = {}            
            for row in res:
                if row[1] not in idx.keys():
                    idx[row[1]] = []
                idx[row[1]].append(row)
            i = 0
            for c in idx.keys():
                if len(idx[c]) < n_min:
                    continue
                if len(idx[c]) > n_max:
                    idx[c] = idx[c][:n_max]
                coins[i] = idx[c]
                i += 1
                
        return coins

    def __get_coin_url(self, filename, path, plaster):
        """
        Based on the filename and path arguments of coin database entries
        the URLs to fetch the images are built
        """
        url_end = "dw=100&dh=100"

        if plaster:
            return ".png", "PNG", base_url + filename + url_end
        else:
            if not path:            
                url = base_url + filename + url_end
            elif "Muenzkabinett" in path:
                url = base_url + "Muenzkabinett/" + filename + url_end
                
            else: url = path + '/' + filename

            return ".jpg", "JPEG", url

    def __cut_edges(self, image, flip):
        """
        Image is flipped along the axes specified in flip argument
        for corresponding edge to be cut
        """
        if flip is not None:
            image = flip(image)
        for i, row in enumerate(image):
            if np.min(row)<0.9:
                cut_off = i
                if cut_off > 0:         
                    image = image[cut_off:]
                break
        if flip is not None:
            if flip is np.rot90:
                image = np.rot90(image, 3)
            else:
                image = flip(image)
        return image

    def __format_image(self, http_result):
        """
        Format images according to the image_params
        """
        try:
            b = BytesIO(http_result.content)
            image = io.imread(b, as_gray=self.image_params.as_gray)
            #thresh = filters.threshold_minimum(image)
        except(Exception):
            print("Oops... Something went wrong while reading HTTP response")
            return
        
        # Cut off edges from images by discerning pixel rows and columns
        # where coin borders are not reached
        if self.image_params.cut_edges:
            flips = [None, np.rot90, np.flip, np.transpose]
            for flip in flips:
                image = self.__cut_edges(image, flip)

        image = transform.resize(np.asarray(image.copy()), (self.image_params.size,self.image_params.size), mode='wrap')
        image = (image*255).astype(np.uint8)   

        return image
                

    def create_dataset(self, data_path='Coinset'):
        """
        This function creates the coin dataset in a folder of structure
        Dataset
            --class x
                --coin x
                --coin y
            --class y
            ...

        Images are fetched according to the path and filename
        and formatted as described in 'image_params'
        """
        if data_path not in os.listdir():
            os.mkdir(data_path)
        for k, v in self.coins.items():
            for coin in v:
                res = None
                filename = coin[2]
                path = coin[4]
                object_type = coin[3]
                if filename is not None:
                    ext, frmt, url = self.__get_coin_url(filename, path, object_type=='plastercast')
                    try:                        
                        res = requests.get(url)
                    except (Exception):
                        print("Oops... Something went wrong while requesting the web resource")
                        continue
                image = self.__format_image(res)
                if image is None:
                    continue
                if (str(k) not in os.listdir(data_path)):
                    os.mkdir(os.join(data_path, str(k))

                try:
                    io.imsave(os.path.join(data_path, str(k), str(coin[0])) + ext, image)
                except UserWarning:
                    continue

