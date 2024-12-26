import mysql.connector
import random,sys
# Database connection setup
# def connect_to_database():
#     return mysql.connector.connect(
#         host="localhost",  # Replace with your database host
#         user="root",  # Replace with your username
#         password="ircadircad",  # Replace with your password
#         database="snipr_results"  # Replace with your database name
#     )
#

# Database connection setup
def connect_to_database():
    # Pool of random IP addresses
    ip_pool = [
        "127.0.0.1",
        "192.168.1.100",  # Replace with your database server IPs
        "192.168.1.101",
        "203.0.113.50",
        "203.0.113.51"
    ]

    # Randomly select an IP address from the pool
    random_ip = random.choice(ip_pool)

    try:
        # Establish connection
        connection = mysql.connector.connect(
            host='10.39.217.11', ###'128.252.210.4', ##'10.39.217.11', #random_ip,  # Use the randomly selected IP
            user="root",  # Replace with your username
            password="ircadircad",  # Replace with your password
            database="snipr_results"  # Replace with your database name
        )
        print(f"Connected to database at {random_ip}")
        return connection
    except mysql.connector.Error as error:
        print(f"Failed to connect to the database: {error}")
        return None

# Insert data into the table
def insert_data(session_id, session_name, scan_id, scan_name):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # SQL query to insert data

        sql_query = """
            INSERT INTO results (session_id, session_name, scan_id, scan_name,session_id_scan_id)
            VALUES (%s, %s, %s, %s,%s)
        """

        # Parameters for the query
        data = (session_id, session_name, scan_id, scan_name,str(session_id)+"_"+str(scan_id))

        # Execute and commit the transaction
        cursor.execute(sql_query, data)
        connection.commit()


        print(f"Record inserted successfully. ID: {cursor.lastrowid}")

    except mysql.connector.Error as error:
        print(f"Failed to insert record into table: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

# Function to check if a column exists
def column_exists(cursor, table_name, column_name):
    cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE '{column_name}';")
    return cursor.fetchone() is not None

# Main function to update or create column
def update_or_create_column(session_id, scan_id, column_name, column_value,session_name="SESSION_NAME",scan_name="SCAN_NAME"):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Step 1: Generate session_id_scan_id value
        session_id_scan_id = f"{session_id}_{scan_id}"
        if not column_exists(cursor, "results", 'session_id_scan_id'):
            insert_data(session_id, session_name, scan_id, scan_name)
        # Add the new column if it doesn't exist
        #     alter_query = f"ALTER TABLE results ADD COLUMN {column_name} VARCHAR(255);"
        #     cursor.execute(alter_query)
        #     connection.commit()
        #     print(f"Column '{column_name}' added as VARCHAR(255).")
        # Step 2: Check if the row exists based on session_id_scan_id
        select_query = """
            SELECT * FROM results
            WHERE session_id_scan_id = %s;
        """
        cursor.execute(select_query, (session_id_scan_id,))
        row = cursor.fetchone()  # Fetch one row to consume the result

        if row is None:
            print("No matching row found. Inserting new row...")
            # Insert a new row if it doesn't exist
            insert_query = """
                INSERT INTO results (session_id_scan_id, session_id, scan_id)
                VALUES (%s, %s, %s);
            """
            cursor.execute(insert_query, (session_id_scan_id, session_id, scan_id))
            connection.commit()

        # Step 3: Check if the new column exists
        if not column_exists(cursor, "results", column_name):
            # Add the new column if it doesn't exist
            alter_query = f"ALTER TABLE results ADD COLUMN {column_name} VARCHAR(255);"
            cursor.execute(alter_query)
            connection.commit()
            print(f"Column '{column_name}' added as VARCHAR(255).")

        # Step 4: Update the new column with the provided value for the matching row
        update_query = f"""
            UPDATE results
            SET {column_name} = %s
            WHERE session_id_scan_id = %s;
        """
        cursor.execute(update_query, (column_value, session_id_scan_id))
        connection.commit()

        print(f"'{column_name}' updated to '{column_value}' for session_id='{session_id}' and scan_id='{scan_id}'.")

    except mysql.connector.Error as error:
        print(f"Failed to execute operation: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

# Example usage
if __name__ == "__main__":
    # Replace these with actual values
    session_id = sys.argv[1] # "session_133"
    session_name =sys.argv[2] #  "Session Name Example"
    scan_id =sys.argv[3] #  "scan_456"
    scan_name =sys.argv[4] #  "Scan Name Example"

    # Insert initial data
    insert_data(session_id, session_name, scan_id, scan_name)

    # Update or create column
    column_name =sys.argv[5] #  "volume"  # Specify the new column name
    column_value =sys.argv[6] #  "200"  # Value to be set in the new column

    update_or_create_column(session_id, scan_id, column_name, column_value,session_name,scan_name)
    # update_or_create_column(session_id, scan_id, 'session_name', session_name,session_name,scan_name)
    # update_or_create_column(session_id, scan_id, 'scan_name', scan_name,session_name,scan_name)
