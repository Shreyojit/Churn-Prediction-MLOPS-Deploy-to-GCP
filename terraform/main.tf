data "google_container_engine_versions" "default" {
  location = var.zone
}

data "google_client_config" "current" {}

resource "google_container_cluster" "default" {
  name               = "fraud-detection-cluster"
  location           = var.zone
  initial_node_count = 3
  min_master_version = data.google_container_engine_versions.default.latest_master_version

  node_config {
    machine_type = "e2-medium" # Use a cost-effective machine type
    disk_size_gb = 32
  }

  provisioner "local-exec" {
    when    = destroy
    command = "sleep 90"
  }
}