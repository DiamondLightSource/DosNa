

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>


#include "clovis/clovis.h"
#include "clovis/clovis_idx.h"

#include "clovis_functions.h"


static struct m0_clovis *clovis_instance = NULL;
static struct m0_clovis_container clovis_container;
static struct m0_clovis_realm clovis_uber_realm;
static struct m0_clovis_config clovis_conf;
static struct m0_idx_dix_config dix_conf;

static size_t   clovis_block_size = 4096;

extern struct m0_addb_ctx m0_clovis_addb_ctx;


int
init_clovis(char *laddr, char *ha_addr, char *prof_id, char *proc_fid,
	    size_t block_size)
{
	int             rc;

	dix_conf = (struct m0_idx_dix_config) {
		.kc_create_meta = false
	};

	clovis_conf = (struct m0_clovis_config) {
		.cc_local_addr = laddr,
		.cc_ha_addr = ha_addr,
		.cc_profile = prof_id,
		.cc_process_fid = proc_fid,
		.cc_is_oostore = true,
		.cc_is_read_verify = false,
		.cc_max_rpc_msg_size = M0_RPC_DEF_MAX_RPC_MSG_SIZE,
		.cc_tm_recv_queue_min_len = M0_NET_TM_RECV_QUEUE_DEF_LEN,
		.cc_idx_service_id = M0_CLOVIS_IDX_DIX,
		.cc_idx_service_conf = &dix_conf
	};

	clovis_block_size = block_size;

	rc = m0_clovis_init(&clovis_instance, &clovis_conf, true);

	if (rc)
		return rc;

	m0_clovis_container_init(&clovis_container, NULL, &M0_CLOVIS_UBER_REALM,
				 clovis_instance);

	rc = clovis_container.co_realm.re_entity.en_sm.sm_rc;

	if (rc != 0)
		goto init_error1;

	clovis_uber_realm = clovis_container.co_realm;
	return 0;

init_error1:
	m0_clovis_fini(clovis_instance, true);
	return rc;
}

void
fini_clovis()
{
	m0_clovis_fini(clovis_instance, true);
}

static int
open_entity(struct m0_clovis_entity *entity)
{
	int             rc = 0;
	struct m0_clovis_op *ops[1] = { NULL };

	m0_clovis_entity_open(entity, &ops[0]);
	m0_clovis_op_launch(ops, 1);
	m0_clovis_op_wait(ops[0],
					  M0_BITS(M0_CLOVIS_OS_FAILED, M0_CLOVIS_OS_STABLE),
			                  M0_TIME_NEVER);
	// this return code is not the state machine return code
	rc = m0_clovis_rc(ops[0]);
	m0_clovis_op_fini(ops[0]);
	m0_clovis_op_free(ops[0]);
	return rc;
}

static int
delete_entity(struct m0_clovis_entity *entity)
{
	int             rc = 0;
	struct m0_clovis_op *ops[1] = { NULL };

	m0_clovis_entity_delete(entity, &ops[0]);
	m0_clovis_op_launch(ops, 1);
	rc = m0_clovis_op_wait(ops[0], M0_BITS(M0_CLOVIS_OS_FAILED,
					       M0_CLOVIS_OS_STABLE), M0_TIME_NEVER);
	m0_clovis_op_fini(ops[0]);
	m0_clovis_op_free(ops[0]);
	return rc;
}

static int
write_data_to_object(struct m0_uint128 id, struct m0_indexvec *ext,
		     struct m0_bufvec *data, struct m0_bufvec *attr)
{
	int             rc = 0;
	struct m0_clovis_obj obj;
	struct m0_clovis_op *ops[1] = { NULL };

	memset(&obj, 0, sizeof(struct m0_clovis_obj));

	m0_clovis_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_clovis_layout_id(clovis_instance));

	open_entity(&obj.ob_entity);
	m0_clovis_obj_op(&obj, M0_CLOVIS_OC_WRITE, ext, data, attr, 0, &ops[0]);

	m0_clovis_op_launch(ops, 1);

	rc = m0_clovis_op_wait(ops[0], M0_BITS(M0_CLOVIS_OS_FAILED,
					       M0_CLOVIS_OS_STABLE), M0_TIME_NEVER);

	if (ops[0]->op_sm.sm_state != M0_CLOVIS_OS_STABLE
	    || ops[0]->op_sm.sm_rc != 0)
		rc = EPERM;

	m0_clovis_op_fini(ops[0]);
	m0_clovis_op_free(ops[0]);

	m0_clovis_entity_fini(&obj.ob_entity);
	return rc;
}

static int
read_data_from_object(struct m0_uint128 id, struct m0_indexvec *ext,
		      struct m0_bufvec *data, struct m0_bufvec *attr)
{
	int             rc = 0;
	struct m0_clovis_obj obj;
	struct m0_clovis_op *ops[1] = { NULL };

	memset(&obj, 0, sizeof(struct m0_clovis_obj));

	m0_clovis_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_clovis_layout_id(clovis_instance));

	rc = open_entity(&obj.ob_entity);
	if (rc < 0)		// object not found
		return rc;

	m0_clovis_obj_op(&obj, M0_CLOVIS_OC_READ, ext, data, attr, 0, &ops[0]);

	if (ops[0] == NULL || ops[0]->op_sm.sm_rc != 0) {
		rc = EPERM;
		goto read_exit1;
	}

	m0_clovis_op_launch(ops, 1);

	rc = m0_clovis_op_wait(ops[0],
			       M0_BITS(M0_CLOVIS_OS_FAILED, M0_CLOVIS_OS_STABLE),
			       M0_TIME_NEVER);

	if (ops[0]->op_sm.sm_state != M0_CLOVIS_OS_STABLE
	    || ops[0]->op_sm.sm_rc != 0)
		rc = -EPERM;

	m0_clovis_op_fini(ops[0]);
	m0_clovis_op_free(ops[0]);
read_exit1:
	m0_clovis_entity_fini(&obj.ob_entity);
	return rc;
}

int
create_object(uint64_t high_id, uint64_t low_id)
{
	int             rc = 0;
	struct m0_clovis_obj obj;
	struct m0_clovis_op *ops[1] = { NULL };

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	memset(&obj, 0, sizeof(struct m0_clovis_obj));

	m0_clovis_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_clovis_layout_id(clovis_instance));

	rc = open_entity(&obj.ob_entity);
	if (rc >= 0)		// object already exists
		return 1;

	m0_clovis_entity_create(NULL, &obj.ob_entity, &ops[0]);
	m0_clovis_op_launch(ops, ARRAY_SIZE(ops));

	rc = m0_clovis_op_wait(ops[0],
			       M0_BITS(M0_CLOVIS_OS_FAILED, M0_CLOVIS_OS_STABLE),
			       M0_TIME_NEVER);
	m0_clovis_op_fini(ops[0]);
	m0_clovis_op_free(ops[0]);
	m0_clovis_entity_fini(&obj.ob_entity);

	return rc;
}

int
delete_object(uint64_t high_id, uint64_t low_id)
{
	int             rc = 0;
	struct m0_clovis_obj obj;

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	memset(&obj, 0, sizeof(struct m0_clovis_obj));

	m0_clovis_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_clovis_layout_id(clovis_instance));

	rc = open_entity(&obj.ob_entity);
	if (rc < 0) // object not found
		return rc;

	rc = delete_entity(&obj.ob_entity);

	m0_clovis_entity_fini(&obj.ob_entity);

	return rc;
}

/*
 * read the data contained in the object with the id specified and
 * write it to buffer.
 * it returns 0 if the operation was correct
 */
int
read_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length)
{
	int             rc = 0;

	int             n_full_blocks = length / clovis_block_size;
	int             n_blocks = n_full_blocks + (length % clovis_block_size != 0);

	struct m0_indexvec ext;
	struct m0_bufvec data;
	struct m0_bufvec attr;
	int             i;

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	rc = m0_indexvec_alloc(&ext, n_blocks);
	if (rc)
		return rc;

	rc = m0_bufvec_alloc(&data, n_blocks, clovis_block_size);
	if (rc)
		goto read_exit2;

	rc = m0_bufvec_alloc(&attr, n_blocks, 1);
	if (rc)
		goto read_exit1;

	int             byte_count = 0;
	for (i = 0; i < n_blocks; i++) {
		ext.iv_index[i] = byte_count;
		ext.iv_vec.v_count[i] = clovis_block_size;
		byte_count += clovis_block_size;
		attr.ov_vec.v_count[i] = 0;
	}

	rc = read_data_from_object(id, &ext, &data, &attr);
	if (rc)
		goto read_exit;

	for (i = 0; i < n_full_blocks; i++) {
		memcpy(buffer + ext.iv_index[i],
			   data.ov_buf[i],
			   clovis_block_size);
	}

	if (n_blocks > n_full_blocks) {
		memcpy(buffer + n_full_blocks * clovis_block_size,
		       data.ov_buf[n_full_blocks],
			   length % clovis_block_size);
	}

read_exit:
	m0_bufvec_free(&attr);
read_exit1:
	m0_bufvec_free(&data);
read_exit2:
	m0_indexvec_free(&ext);

	return rc;
}

/*
 * write data from buffer to object specified by the id.
 * It returns 0 if the operation was correct.
 */
int
write_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length)
{
	int             rc = 0;
	int             n_full_blocks = length / clovis_block_size;
	int             n_blocks = n_full_blocks + (length % clovis_block_size != 0 ? 1 : 0);
	struct m0_indexvec ext;
	struct m0_bufvec data;
	struct m0_bufvec attr;
	int             i;

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	rc = m0_indexvec_alloc(&ext, n_blocks);
	if (rc)
		return rc;
	rc = m0_bufvec_alloc(&data, n_blocks, clovis_block_size);
	if (rc)
		goto write_exit2;
	rc = m0_bufvec_alloc(&attr, n_blocks, 1);
	if (rc)
		goto write_exit1;
	int             byte_count = 0;

	for (i = 0; i < n_full_blocks; i++) {
		ext.iv_index[i] = byte_count;
		ext.iv_vec.v_count[i] = clovis_block_size;
		byte_count += clovis_block_size;
		attr.ov_vec.v_count[i] = 0;
		memcpy(data.ov_buf[i],
			   buffer + i * clovis_block_size,
			   clovis_block_size);
	}

	if (n_blocks > n_full_blocks) {
		ext.iv_index[n_full_blocks] = byte_count;
		ext.iv_vec.v_count[n_full_blocks] = clovis_block_size;
		attr.ov_vec.v_count[n_full_blocks] = 0;
		memcpy(data.ov_buf[n_full_blocks],
		       buffer + n_full_blocks * clovis_block_size,
		       length % clovis_block_size);
	}
	rc = write_data_to_object(id, &ext, &data, &attr);

	m0_bufvec_free(&attr);
write_exit1:
	m0_bufvec_free(&data);
write_exit2:
	m0_indexvec_free(&ext);
	return rc;
}

int
exist_object(uint64_t high_id, uint64_t low_id)
{
	int             rc;
	struct m0_clovis_obj obj;
	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	memset(&obj, 0, sizeof(struct m0_clovis_obj));

	m0_clovis_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_clovis_layout_id(clovis_instance));
	rc = open_entity(&obj.ob_entity);

	m0_clovis_entity_fini(&obj.ob_entity);
	return rc >= 0;
}
