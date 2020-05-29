from abc import ABC, abstractmethod


class AbstractBannerReplacer(ABC):
    @abstractmethod
    def build_model(self, filename):
        pass

    @abstractmethod
    def detect_banner(self):
        pass

    @abstractmethod
    def insert_logo(self):
        pass
