# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)


import sys
import sqlite3
import numpy as np
import os


IS_PYTHON3 = sys.version_info[0] >= 3


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def dbReturnMatches(database_path, min_num_matches):
    print(database_path)
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = {}
        matches = {}
        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name

        cursor.execute("SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;", (min_num_matches,))

        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1]
            image_name2 = images[image_id2]
            matches["{} {}".format(image_name1, image_name2)] = inlier_matches

        cursor.close()
        connection.close()

        return matches

    else:
        print("Database does not exist")
        quit()