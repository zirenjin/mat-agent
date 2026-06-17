import site, os, glob, shutil
sitepkg = site.getsitepackages()[0]
dist_info = os.path.join(sitepkg, "mpich-0.0.0.dist-info")
os.makedirs(dist_info, exist_ok=True)

# Copy real MPI .so files to site-packages root (where .locate() will look)
real_libs = []
for lib in sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libmpich*.so*")):
    real = os.path.realpath(lib)
    real_name = os.path.basename(real)
    dst = os.path.join(sitepkg, real_name)  # put in site-packages root
    if not os.path.exists(dst):
        shutil.copy2(real, dst)
        real_libs.append(real_name)
        print(f"Copied {real_name}")

# Create libmpi* symlinks in site-packages root
names = list(real_libs)
for name in real_libs:
    generic = name.replace("libmpich", "libmpi")
    gen_dst = os.path.join(sitepkg, generic)
    if not os.path.lexists(gen_dst):
        os.symlink(name, gen_dst)
        names.append(generic)
        print(f"Symlink {generic} -> {name}")

# Metadata in dist-info
with open(os.path.join(dist_info, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: mpich\nVersion: 0.0.0\n")

# RECORD entries relative to site-packages (the distribution location)
with open(os.path.join(dist_info, "RECORD"), "w") as f:
    for name in sorted(set(names)):
        f.write(f"{name},,\n")
    # dist-info files relative to site-packages
    f.write("mpich-0.0.0.dist-info/METADATA,,\n")
    f.write("mpich-0.0.0.dist-info/RECORD,,\n")

with open(os.path.join(dist_info, "INSTALLER"), "w") as f:
    f.write("pip\n")

print(f"Files in site-packages: {sorted(set(names))}")
