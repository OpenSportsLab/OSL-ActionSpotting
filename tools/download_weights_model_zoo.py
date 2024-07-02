
import urllib.request
import os
from tqdm import tqdm
import json
import random
from SoccerNet.utils import getListGames

class MyProgressBar():
    def __init__(self, filename):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            self.pbar.set_description(f"Downloading {self.filename}...")
            self.pbar.refresh()  # to show immediately the update

        self.pbar.update(block_size)


class OwnCloudDownloader():
    def __init__(self, LocalDirectory, 
                 OwnCloudServer="https://exrcsdrive.kaust.edu.sa/public.php/webdav/"):
        self.LocalDirectory = LocalDirectory
        self.OwnCloudServer = OwnCloudServer

    def downloadFile(self, path_local, path_owncloud, user=None, password=None, verbose=True):
        # return 0: successfully downloaded
        # return 1: HTTPError
        # return 2: unsupported error
        # return 3: file already exist locally
        # return 4: password is None
        # return 5: user is None

        if password is None:
            print(f"password required for {path_local}")
            return 4
        if user is None:
            return 5

        if user is not None or password is not None:  
            # update Password
             
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(
                None, self.OwnCloudServer, user, password)
            handler = urllib.request.HTTPBasicAuthHandler(
                password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)

        if os.path.exists(path_local): # check existence
            if verbose:
                print(f"{path_local} already exists")
            return 2

        try:
            try:
                os.makedirs(os.path.dirname(path_local), exist_ok=True)
                urllib.request.urlretrieve(
                    path_owncloud, path_local, MyProgressBar(path_local))

            except urllib.error.HTTPError as identifier:
                print(identifier)
                return 1
        except:
            if os.path.exists(path_local):
                os.remove(path_local)
            raise
            return 2


        return 0



class ModelZooDownloader(OwnCloudDownloader):
    def __init__(self, LocalDirectory,
                 OwnCloudServer="https://exrcsdrive.kaust.edu.sa/public.php/webdav/"):
        super(ModelZooDownloader, self).__init__(
            LocalDirectory, OwnCloudServer)
        self.password = "OSL"
    
    def downloadWeights(self, model, file):

        FileLocal = os.path.join(self.LocalDirectory, model, file)
        FileURL = os.path.join(self.OwnCloudServer, model, file).replace(' ', '%20').replace('\\', '/')
        res = self.downloadFile(path_local=FileLocal,
                                path_owncloud=FileURL,
                                user="eIjTapzHicsb4yy",  
                                password="OSL")
        if res == 0: print("successfully downloaded")
        if res == 1: print("HTTPError")
        if res == 2: print("unsupported error")
        if res == 3: print("file already exist locally")
        if res == 4: print("Password is wrong or None")
        if res == 5: print("User is wrong or None")


                    
if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # Load the arguments
    parser = ArgumentParser(description='Test Downloader',
                            formatter_class=ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--Model',   required=True,
                        type=str, help='Model to download')
    parser.add_argument('--File',   required=False,
                        type=str, help='File to download')
    args = parser.parse_args()

    myModelZooDownloader = ModelZooDownloader(LocalDirectory="models")
    
    myModelZooDownloader.downloadWeights(args.Model, args.File)
   