from networktables import NetworkTables


class NetworkTablesClient:

    def __init__(self, server, device_name):
        self.network_tables = NetworkTables.initialize(server=server)
        self.smartdashboard = NetworkTables.getTable(device_name)

    def putBoolean(self, key, value):
        self.smartdashboard.putBoolean(key, value)

    def putNumberArray(self, key, value):
        self.smartdashboard.putNumber(key, valu)

    def putNumberArray(self, key, values):
        self.smartdashboard.putNumberArray(key, values)

    def putString(self, key, value):
        self.smartdashboard.putString(key, value)