#!/bin/bash
echo "Installing InfluxDB locally for development..."
curl -L -o install_influxdb2.sh https://www.influxdata.com/d/install_influxdb2.sh
sudo sh install_influxdb2.sh enterprise
rm install_influxdb2.sh
echo "InfluxDB installed. Start with: sudo systemctl start influxdb"
echo "Access InfluxDB UI at: http://localhost:8086"
echo "To enable auto-start: sudo systemctl enable influxdb"