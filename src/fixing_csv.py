import csv

src_path = "data/products backup.csv"              # CSV lama: delimiter koma
dst_path = "data/product_semicolon.csv"    # CSV baru: delimiter ; 

with open(src_path, "r", encoding="utf-8", newline="") as f_in, \
     open(dst_path, "w", encoding="utf-8", newline="") as f_out:
    
    # reader akan mengerti koma di dalam "..." itu bukan delimiter
    reader = csv.reader(f_in, delimiter=",")
    
    # writer akan menulis dengan delimiter ; dan menambahkan quote jika perlu
    writer = csv.writer(f_out, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    
    for row in reader:
        writer.writerow(row)

print(f"✅ File baru tersimpan di: {dst_path}")
