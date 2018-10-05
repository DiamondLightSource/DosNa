#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <clovis_functions.h>

#define LINE_BUF_SIZE 256


static size_t clovis_block_size = 4096;

char clovis_local_addr[LINE_BUF_SIZE];
char clovis_ha_addr[LINE_BUF_SIZE];
char clovis_prof[LINE_BUF_SIZE];
char clovis_proc_fid[LINE_BUF_SIZE];
char rc_filename[] = "sagerc";
unsigned int tier = 0;

int test_write_and_read_object() {
	char *buff;
	char data[] = "abcdefghijklmnopqrstuvwxyz";
	size_t data_len = strlen(data);
	buff = (char *) malloc(data_len + 1);
	uint64_t high_id = 10;
	uint64_t low_id = 16;
	int rc;
	printf("Creating ...\n");
	rc = create_object(high_id, low_id);
	if (rc) goto end1;
	printf("Setting ...\n");
	rc = write_object(high_id, low_id, data, data_len);
	if (rc) goto end1;
	printf("Reading ...\n");
	rc = read_object(high_id, low_id, buff, data_len);
	if (rc) goto end1;
	assert(strncmp(data, buff, data_len) == 0);
	buff[data_len] = 0;
	printf("Got ... %s\n", buff);
end1:
	printf("exit rc: %d\n", rc);
	printf("Deleting ...\n");
	rc = delete_object(high_id, low_id);
	assert(rc == 0);
	free(buff);
	return 0;
}

int test_exist_object() {
	int exists;
	uint64_t high_id = 1;
	uint64_t low_id = 14;
	printf("Checking existence without creating object\n");
	exists = exist_object(high_id, low_id);
	assert(!exists);
	create_object(high_id, low_id);
	printf("Checking existence after creating an object\n");
	exists = exist_object(high_id, low_id);
	assert(exists);
	delete_object(high_id, low_id);
	printf("Checking existence after deleting an object\n");
	exists = exist_object(high_id, low_id);
	assert(!exists);
	return 0;
}

int get_line(FILE *fp, char *dst) {
	char buffer[LINE_BUF_SIZE] = {0};
	while (1) {
		assert(fgets(buffer, LINE_BUF_SIZE, fp) != NULL);
		if (strlen(buffer) == 0 || buffer[0] == '#' || buffer[0] == '\n')
			continue;
		char *end = buffer + strlen(buffer) - 1;
		while (*end == '\n')
			*(end--) = '\0';
		strcpy(dst, buffer);
		break;
	}
	return 1;
}

void load_config() {
	FILE *rc_file = fopen(rc_filename, "r");
	assert(rc_file != NULL);
	get_line(rc_file, clovis_local_addr);
	printf("laddr: %s\t", clovis_local_addr);
	get_line(rc_file, clovis_ha_addr);
	printf("ha_addr: %s\t", clovis_ha_addr);
	get_line(rc_file, clovis_prof);
	printf("prof: %s\t", clovis_prof);
	get_line(rc_file, clovis_proc_fid);
	printf("proc fid: %s\n", clovis_proc_fid);
	fclose(rc_file);
}

int main() {
	int rc;
	load_config();
	rc = init_clovis(clovis_local_addr, clovis_ha_addr, clovis_prof,
			clovis_proc_fid, clovis_block_size);
	assert(rc == 0);
	test_write_and_read_object();
	test_exist_object();
	fini_clovis();
	return 0;
}
