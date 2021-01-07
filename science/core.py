import abc
import datetime


# TODO: Come up with a better name
class Science(abc.ABC):
    def __init__(
            self,
            run_datetime: datetime.datetime = datetime.datetime.utcnow(),
            start_date: datetime.datetime = datetime.date(year=2010, month=1, day=1),
            is_prod: bool = False,
            archive_files: bool = False,
    ):
        self.run_datetime = run_datetime
        self.start_date = start_date
        self._is_prod = is_prod
        self._archive_files = archive_files

    @property
    def is_prod(self) -> bool:
        """Whether the model is production or not"""
        return self._is_prod

    @property
    def archive_files(self) -> bool:
        """Whether to save output files"""
        return self._archive_files

    @property
    def location(self) -> str:
        """String representation of whether the model will be trained or not"""
        return 'prod' if self.is_prod else 'dev'

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Model id for filename(s)"""
        pass

    def filename(
            self,
            prefix: str,
    ) -> str:
        """Filename for archiving results"""
        return f"{prefix}_{self.run_datetime.strftime('%Y%m%d%H%M%S')}"

    @property
    @abc.abstractmethod
    def query(self) -> str:
        """Query to retrieve raw data"""
        pass

    @abc.abstractmethod
    def execute(self):
        pass
