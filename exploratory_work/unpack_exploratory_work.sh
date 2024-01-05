if test -f ".clusters_unpacked" ; then
  printf "One time extraction of exploratory work data successfully completed previously, skipping.\n"
else
  printf "Extracting clusters.tar.gz to clusters/.\n"
  printf "Started at $(date)\n"
  tar -xvzf clusters.tar.gz && \
  touch .clusters_unpacked && printf "Finished at $(date)\n"
fi