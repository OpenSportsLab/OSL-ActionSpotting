import urllib.request
import os
from tqdm import tqdm


class MyProgressBar:
    def __init__(self, filename):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit="iB", unit_scale=True)
            self.pbar.set_description(f"Downloading {self.filename}...")
            self.pbar.refresh()  # to show immediately the update

        self.pbar.update(block_size)


import uuid
from google_measurement_protocol import event, report


class OwnCloudDownloader:
    def __init__(self, LocalDirectory, OwnCloudServer):
        self.LocalDirectory = LocalDirectory
        self.OwnCloudServer = OwnCloudServer

        self.client_id = uuid.uuid4()

    def downloadFile(
        self, path_local, path_owncloud, user=None, password=None, verbose=True
    ):
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
            password_mgr.add_password(None, self.OwnCloudServer, user, password)
            handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)

        if os.path.exists(path_local):  # check existence
            if verbose:
                print(f"{path_local} already exists")
            return 2

        try:
            try:
                os.makedirs(os.path.dirname(path_local), exist_ok=True)
                urllib.request.urlretrieve(
                    path_owncloud, path_local, MyProgressBar(path_local)
                )

            except urllib.error.HTTPError as identifier:
                print(identifier)
                return 1
        except:
            if os.path.exists(path_local):
                os.remove(path_local)
            raise
            return 2

        # record googleanalytics event
        data = event("download", os.path.basename(path_owncloud))
        report("UA-99166333-3", self.client_id, data)

        return 0


class SoccerNetDownloader(OwnCloudDownloader):
    def __init__(
        self,
        LocalDirectory,
        OwnCloudServer="https://exrcsdrive.kaust.edu.sa/public.php/webdav/",
    ):
        super(SoccerNetDownloader, self).__init__(LocalDirectory, OwnCloudServer)
        self.password = None

    def downloadConfigTask(self, task, version, config, verbose=True, password="OSL"):
        if task == "spotting-OSL":
            if version == "224p":
                if config == "rny008gsm_150_dali":
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory, task, config, "checkpoint_141.pt"
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "checkpoint_141.pt"
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory, task, config, "e2espot.py"
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "e2espot.py"
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )
                if config == "rny008gsm_150_opencv":
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory, task, config, "checkpoint_141.pt"
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "checkpoint_141.pt"
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory, task, config, "e2espot.py"
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "e2espot.py"
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )
            if config == "rny002gsm_100_dali":
                res = self.downloadFile(
                    path_local=os.path.join(
                        self.LocalDirectory, task, config, "checkpoint_099.pt"
                    ),
                    path_owncloud=os.path.join(
                        self.OwnCloudServer, config, "checkpoint_099.pt"
                    )
                    .replace(" ", "%20")
                    .replace("\\", "/"),
                    user="eIjTapzHicsb4yy",
                    password=password,
                    verbose=verbose,
                )
                res = self.downloadFile(
                    path_local=os.path.join(
                        self.LocalDirectory, task, config, "e2espot.py"
                    ),
                    path_owncloud=os.path.join(
                        self.OwnCloudServer, config, "e2espot.py"
                    )
                    .replace(" ", "%20")
                    .replace("\\", "/"),
                    user="eIjTapzHicsb4yy",
                    password=password,
                    verbose=verbose,
                )
            if config == "rny002gsm_100_opencv":
                res = self.downloadFile(
                    path_local=os.path.join(
                        self.LocalDirectory, task, config, "checkpoint_099.pt"
                    ),
                    path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "checkpoint_099.pt"
                        )
                    .replace(" ", "%20")
                    .replace("\\", "/"),
                    user="eIjTapzHicsb4yy",
                    password=password,
                    verbose=verbose,
                )
                res = self.downloadFile(
                    path_local=os.path.join(
                        self.LocalDirectory, task, config, "e2espot.py"
                    ),
                    path_owncloud=os.path.join(
                        self.OwnCloudServer, config, "e2espot.py"
                    )
                    .replace(" ", "%20")
                    .replace("\\", "/"),
                    user="eIjTapzHicsb4yy",
                    password=password,
                    verbose=verbose,
                )
            if version == "ResNET_PCA512":
                if config in [
                    "avgpool",
                    "avgpool++",
                    "maxpool",
                    "maxpool++",
                    "netrvlad",
                    "netrvlad++",
                    "calf",
                    "netvlad",
                    "netvlad++",
                ]:
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory, task, config, "model.pth.tar"
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer, config, "model.pth.tar"
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )
                    res = self.downloadFile(
                        path_local=os.path.join(
                            self.LocalDirectory,
                            task,
                            config,
                            f"json_{config}_resnetpca512.py",
                        ),
                        path_owncloud=os.path.join(
                            self.OwnCloudServer,
                            config,
                            f"json_{config}_resnetpca512.py",
                        )
                        .replace(" ", "%20")
                        .replace("\\", "/"),
                        user="eIjTapzHicsb4yy",
                        password=password,
                        verbose=verbose,
                    )


# Download SoccerNet in OSL ActionSpotting format

# mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="outputs")
# mySoccerNetDownloader.downloadConfigTask(task="spotting-OSL", config = "rny008gsm_150_dali", version = "224p")
# mySoccerNetDownloader.downloadConfigTask(task="spotting-OSL", config = "rny008gsm_150_opencv", version = "224p")
# mySoccerNetDownloader.downloadConfigTask(task="spotting-OSL", config = "rny002gsm_100_opencv", version = "224p")
# mySoccerNetDownloader.downloadConfigTask(task="spotting-OSL", config = "rny002gsm_100_dali", version = "224p")
# for config in ["avgpool","avgpool++","maxpool","maxpool++","netrvlad","netrvlad++","calf","netvlad","netvlad++"]:
#     mySoccerNetDownloader.downloadConfigTask(task="spotting-OSL", config = config, version = "ResNET_PCA512")