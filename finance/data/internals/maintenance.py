from finance.data import sql


class DBMaintenance(sql.SQLRunner):

    @property
    def schema_name(self) -> str:
        pass

    @property
    def table_name(self) -> str:
        pass

    @property
    def table_ddl(self) -> str:
        pass

    @property
    def sql_script(self) -> str:
        query = '''
            vacuum;
            '''
        return query

    @property
    def is_maintenance(self) -> bool:
        return True


if __name__ == '__main__':
    DBMaintenance().execute()
