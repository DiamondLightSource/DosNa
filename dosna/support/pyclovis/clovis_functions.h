

#ifndef CLOVIS_FUNCTIONS_H_
#define CLOVIS_FUNCTIONS_H_

#include <stddef.h>

typedef unsigned long uint64_t;

int init_clovis(char *laddr, char *ha_addr, char *prof_id, char *proc_fid, size_t block_size, unsigned int tier);

void fini_clovis(void);

int create_object(uint64_t high_id, uint64_t low_id);

int delete_object(uint64_t high_id, uint64_t low_id);

int read_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length);

int write_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length);

int exist_object(uint64_t high_id, uint64_t low_id);

#endif /* CLOVIS_FUNCTIONS_H_ */
