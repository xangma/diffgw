#include <cstdint>
#include "aligned_nodegroup.decl.h"

class AlignedNodeGroup : public CBase_AlignedNodeGroup {
 public:
  // This member makes the complete chare over-aligned, so the runtime must
  // return storage suitable for alignof(AlignedNodeGroup).
  alignas(64) unsigned char payload[64];

  AlignedNodeGroup(CProxy_Main main) {
    // Volatile prevents an optimizer from assuming the placement was aligned.
    const volatile auto address = reinterpret_cast<std::uintptr_t>(this);
    CkPrintf("AlignedNodeGroup this=%p mod64=%llu alignof=%llu\n",
             static_cast<void*>(this),
             static_cast<unsigned long long>(address % 64),
             static_cast<unsigned long long>(alignof(AlignedNodeGroup)));
    if (address % 64 != 0) CkAbort("Under-aligned nodegroup allocation");
    // Touch the over-aligned member to make an invalid placement observable.
    payload[0] = 0x5a;
    // Queue completion only after this constructor has returned.  Main then
    // exits from its entry method, so construction cannot be skipped.
    main.nodegroupReady();
  }
};

static_assert(alignof(AlignedNodeGroup) >= 64,
              "test must exercise over-aligned nodegroup allocation");

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* msg) {
    delete msg;
    for (int i = 0; i < 16; ++i) CProxy_AlignedNodeGroup::ckNew(thisProxy);
  }

  void nodegroupReady() {
    if (++ready_ == 16 * CkNumNodes()) CkExit();
  }

 private:
  int ready_ = 0;
};

#include "aligned_nodegroup.def.h"
